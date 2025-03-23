import sys
import os

# Ajoutez le chemin du répertoire parent au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import os
import glob
import PyPDF2
from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

class Logger:
    """Class for handling logging configuration and operations."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize the logger with a name and level."""
        self.logger = logging.getLogger(name)   
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)


class PDFDocumentProcessor:
    """Class for processing PDF documents."""
    
    def __init__(self, documents_dir: str, logger: Logger):
        """Initialize with documents directory and logger."""
        self.documents_dir = documents_dir
        self.logger = logger
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    try:
                        text += reader.pages[page_num].extract_text() + "\n"
                    except Exception as e:
                        self.logger.error(f"Error extracting page {page_num} from {pdf_path}: {e}")
                return text
        except Exception as e:
            self.logger.error(f"Error opening file {pdf_path}: {e}")
            return ""
    
    def get_documents(self) -> List[Document]:
        """Load all PDF documents from the directory."""
        self.logger.info(f"Loading documents from {self.documents_dir}")
        
        # Check if directory exists
        if not os.path.exists(self.documents_dir):
            self.logger.error(f"Directory {self.documents_dir} does not exist.")
            return []
        
        # Get list of PDF files
        pdf_files = glob.glob(os.path.join(self.documents_dir, "**/*.pdf"), recursive=True)
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.documents_dir}")
            return []
        
        self.logger.info(f"PDF files found: {len(pdf_files)}")
        
        # Extract text from each PDF and create LlamaIndex documents
        llamaindex_docs = []
        
        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Processing file: {os.path.basename(pdf_file)}")
                # Extract text from PDF
                text = self.extract_text_from_pdf(pdf_file)
                
                if not text:
                    self.logger.warning(f"No text extracted from {pdf_file}, file skipped.")
                    continue
                    
                self.logger.info(f"Text extracted: {len(text)} characters")
                
                # Create a LlamaIndex document
                metadata = {"source": pdf_file, "filename": os.path.basename(pdf_file)}
                llamaindex_docs.append(Document(text=text, metadata=metadata))
                
            except Exception as e:
                self.logger.error(f"Error processing file {pdf_file}: {str(e)}")
        
        self.logger.info(f"Total documents created: {len(llamaindex_docs)}")
        return llamaindex_docs


class LlamaIndexConfigurator:
    """Class for configuring LlamaIndex settings."""
    
    def __init__(self, 
                 embedding_model: str, 
                 chunk_size: int, 
                 chunk_overlap: int, 
                 logger: Logger):
        """Initialize with embedding model, chunk parameters, and logger."""
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
    
    def configure_settings(self):
        """Configure global LlamaIndex settings without OpenAI dependencies."""
        self.logger.info(f"Configuring LlamaIndex settings with HuggingFace only")
        
        # Use HuggingFace for embeddings (no OpenAI dependency)
        embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
        
        # Configure text splitter for chunking
        node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Configure global settings
        Settings.embed_model = embed_model
        Settings.node_parser = node_parser
        
        # Important: disable implicit use of OpenAI
        Settings.llm = None
        
        self.logger.info(f"LlamaIndex settings configured without OpenAI dependencies")


# Dans la classe DocumentIndexer, modifiez la méthode index_documents
class DocumentIndexer:
    """Class for indexing documents using LlamaIndex and Qdrant."""
    
    def __init__(self, 
                 qdrant_path: str, 
                 collection_name: str, 
                 logger: Logger):
        """Initialize with Qdrant path, collection name, and logger."""
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.logger = logger
    
    def index_documents(self, documents: List[Document], nodes=None, embedding_dim=None) -> Optional[VectorStoreIndex]:
        """Index the provided documents or nodes."""
        from uuid import uuid4
        import numpy as np
        
        try:
            if not documents:
                self.logger.error("No documents to index")
                return None
            
            # Récupérer ou générer les nodes
            if nodes is None:
                node_parser = Settings.node_parser
                nodes = node_parser.get_nodes_from_documents(documents)
            
            self.logger.info(f"Preparing to index {len(nodes)} nodes")
            
            # Initialize Qdrant client
            self.logger.info(f"Initializing Qdrant client at {self.qdrant_path}")
            client = QdrantClient(path=self.qdrant_path)
            
            # Forcer la recréation de la collection
            collections = client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name in collection_names:
                self.logger.info(f"Removing existing collection '{self.collection_name}'")
                client.delete_collection(collection_name=self.collection_name)
            
            # Déterminer la dimension du vecteur d'embedding
            if embedding_dim is None:
                embedding_dim = Settings.embed_model.get_text_embedding_dimension()
                self.logger.info(f"Detected embedding dimension: {embedding_dim}")
            
            # Créer une nouvelle collection avec les paramètres corrects
            self.logger.info(f"Creating collection '{self.collection_name}' with dimension {embedding_dim}")
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )
            
            # Générer les embeddings et insérer directement dans Qdrant
            self.logger.info("Generating embeddings and inserting points manually")
            
            # Préparer les données pour l'insertion par lots
            ids = []
            embeddings = []
            payloads = []
            
            embed_model = Settings.embed_model
            
            for i, node in enumerate(nodes):
                self.logger.info(f"Processing node {i+1}/{len(nodes)}")
                
                # Générer l'embedding
                embedding = embed_model.get_text_embedding(node.text)
                
                # Préparer les données pour l'insertion
                node_id = str(uuid4())
                ids.append(node_id)
                embeddings.append(embedding)
                
                # Créer un payload avec les métadonnées du nœud
                payload = {
                    "text": node.text,
                    "document_id": node.ref_doc_id if hasattr(node, "ref_doc_id") else f"doc_{i}",
                    "metadata": node.metadata
                }
                payloads.append(payload)
                
                # Afficher les informations de debug
                self.logger.info(f"Node {i+1}: ID={node_id}, Embedding dimension={len(embedding)}")
            
            # Insérer les points par lots
            self.logger.info(f"Inserting {len(ids)} points into Qdrant collection")
            points = [
                PointStruct(
                    id=id_val,
                    vector=embedding,
                    payload=payload
                )
                for id_val, embedding, payload in zip(ids, embeddings, payloads)
            ]

            client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Vérifier que les points ont été insérés
            collection_info = client.get_collection(collection_name=self.collection_name)
            points_count = collection_info.points_count
            self.logger.info(f"Nombre de points dans la collection : {points_count}")
            
            if points_count == 0:
                self.logger.error("⚠️ ERREUR: Aucun point n'a été indexé malgré l'insertion manuelle.")
                return None
            else:
                self.logger.info(f"✅ Documents successfully indexed: {points_count} points created")
                
                # Créer un vector store et un index pour l'utilisation ultérieure
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=self.collection_name,
                )
                
                # Créer un index vide pour l'interface
                index = VectorStoreIndex.from_vector_store(vector_store)
                
                return index
            
        except Exception as e:
            self.logger.error(f"Error during indexing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

class PDFIndexingManager:
    """Main class for managing the PDF indexing process."""
    
    def __init__(self, 
                 documents_dir: str, 
                 qdrant_path: str, 
                 collection_name: str, 
                 embedding_model: str, 
                 chunk_size: int, 
                 chunk_overlap: int):
        """Initialize with all necessary parameters."""
        self.documents_dir = documents_dir
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = Logger("pdf_indexing_manager")
    
    def run(self) -> Optional[VectorStoreIndex]:
        """Run the complete indexing process."""
        try:
            # Configure LlamaIndex
            configurator = LlamaIndexConfigurator(
                embedding_model=self.embedding_model,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                logger=self.logger
            )
            configurator.configure_settings()
            
            # Process documents
            processor = PDFDocumentProcessor(
                documents_dir=self.documents_dir,
                logger=self.logger
            )
            documents = processor.get_documents()
            
            if not documents:
                self.logger.error("No documents found to index. Process aborted.")
                return None
                
            # Vérifier que les documents ont bien été chargés
            self.logger.info(f"Documents loaded: {len(documents)}")
            for i, doc in enumerate(documents):
                self.logger.info(f"Document {i+1}: {len(doc.text)} chars, source: {doc.metadata.get('source', 'unknown')}")
                
            # Générer les nodes directement pour vérifier le processus de chunking
            node_parser = Settings.node_parser or SentenceSplitter(
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            nodes = node_parser.get_nodes_from_documents(documents)
            self.logger.info(f"Number of nodes/chunks created: {len(nodes)}")
            
            # Vérifier que les nodes ont des embeddings
            if len(nodes) > 0:
                try:
                    # Tester la génération d'embedding sur un noeud
                    test_embedding = Settings.embed_model.get_text_embedding(nodes[0].text)
                    embedding_dim = len(test_embedding)
                    self.logger.info(f"Embedding dimension: {embedding_dim}")
                except Exception as e:
                    self.logger.error(f"Error generating embeddings: {str(e)}")
                    return None
            
            # Index documents with improved error handling
            indexer = DocumentIndexer(
                qdrant_path=self.qdrant_path,
                collection_name=self.collection_name,
                logger=self.logger
            )
            
            # Augmenter la verbosité avant l'indexation
            self.logger.info("Starting document indexing process...")
            
            # Passer les nodes pré-générés plutôt que les documents bruts
            index = indexer.index_documents(documents, nodes=nodes, embedding_dim=embedding_dim)
            
            if index:
                self.logger.info("Indexing process completed successfully")
                return index
            else:
                self.logger.error("Indexing process failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in indexing process: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

if __name__ == "__main__":
    # Configuration
    DOCUMENTS_DIR = r"D:\bureau\BD&AI 1\ci2\S2\th_info\cour"
    QDRANT_PATH = r"D:\bureau\BD&AI 1\ci2\S2\tec_veille\mini_projet\local_qdrant_storage"
    COLLECTION_NAME = "asr_docs"
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Create and run the indexing manager
    manager = PDFIndexingManager(
        documents_dir=DOCUMENTS_DIR,
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    manager.run()
