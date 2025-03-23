import logging
import os
import glob
import PyPDF2
from typing import List, Dict, Any
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class LangchainBM25DocumentSearch:
    def __init__(self, documents_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200, top_k: int = 2):
        self.documents_dir = documents_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.logger = self._setup_logger()
        self.retriever = None

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger("langchain_bm25_script")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        self.logger.info(f"Début de l'extraction de texte depuis {pdf_path}")
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    try:
                        text += reader.pages[page_num].extract_text() + "\n"
                    except Exception as e:
                        self.logger.error(f"Erreur lors de l'extraction de la page {page_num} de {pdf_path}: {e}")
                self.logger.info(f"Fin de l'extraction de texte depuis {pdf_path}")
                return text
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ouverture du fichier {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str, source: str) -> List[Document]:
        self.logger.info(f"Début du découpage de texte pour {source}")
        chunks = []

        if not text:
            return chunks

        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_size = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if current_size + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(
                    Document(
                        page_content=current_chunk,
                        metadata={"source": source}
                    )
                )

                words = current_chunk.split()
                overlap_words = words[-int(len(words) * self.chunk_overlap / current_size):] if words else []
                current_chunk = " ".join(overlap_words) + " " if overlap_words else ""
                current_size = len(current_chunk)

            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_size = len(current_chunk)

        if current_chunk:
            chunks.append(
                Document(
                    page_content=current_chunk,
                    metadata={"source": source}
                )
            )

        self.logger.info(f"Fin du découpage de texte pour {source}")
        return chunks

    def get_document_chunks(self) -> List[Document]:
        self.logger.info(f"Début du chargement des documents depuis {self.documents_dir}")

        if not os.path.exists(self.documents_dir):
            self.logger.error(f"Le répertoire {self.documents_dir} n'existe pas.")
            return []

        pdf_files = glob.glob(os.path.join(self.documents_dir, "**/*.pdf"), recursive=True)

        if not pdf_files:
            self.logger.warning(f"Aucun fichier PDF trouvé dans {self.documents_dir}")
            return []

        self.logger.info(f"Fichiers PDF trouvés: {len(pdf_files)}")
        all_chunks = []

        for pdf_file in pdf_files:
            try:
                self.logger.info(f"Traitement du fichier: {os.path.basename(pdf_file)}")
                text = self.extract_text_from_pdf(pdf_file)

                if not text:
                    self.logger.warning(f"Aucun texte extrait de {pdf_file}, fichier ignoré.")
                    continue

                self.logger.info(f"Texte extrait: {len(text)} caractères")
                chunks = self.chunk_text(text, pdf_file)
                self.logger.info(f"Chunks créés: {len(chunks)}")
                all_chunks.extend(chunks)

            except Exception as e:
                self.logger.error(f"Erreur lors du traitement du fichier {pdf_file}: {str(e)}")

        self.logger.info(f"Nombre total de chunks créés: {len(all_chunks)}")
        self.logger.info(f"Fin du chargement des documents depuis {self.documents_dir}")
        return all_chunks

    def initialize_bm25_retriever(self, documents: List[Document]):
        self.logger.info("Début de l'initialisation du BM25Retriever de Langchain")
        
        # Création du retriever BM25 avec les documents
        self.retriever = BM25Retriever.from_documents(
            documents, 
            k=self.top_k
        )
        
        self.logger.info(f"BM25Retriever initialisé avec {len(documents)} documents")
        return self.retriever

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Début de la recherche de la requête: '{query}'")
        
        if not self.retriever:
            self.logger.error("Le retriever n'a pas été initialisé.")
            return []
            
        documents = self.retriever.get_relevant_documents(query)
        self.logger.info(f"{len(documents)} documents trouvés pour la requête")
        results = []

        for i, doc in enumerate(documents):
            results.append({
                "rank": i + 1,
                "score": getattr(doc, 'score', 'N/A'),  # Certains retrievers ne retournent pas de score
                "content": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "document_id": i  # Langchain n'attribue pas automatiquement d'ID
            })

        self.logger.info(f"Fin de la recherche de la requête: '{query}'")
        return results

    def main(self):
        self.logger.info("Début de l'exécution du script principal")
        try:
            chunks = self.get_document_chunks()

            if not chunks:
                self.logger.error("Aucun document trouvé, impossible de continuer.")
                return

            self.initialize_bm25_retriever(chunks)
            print("\n=== Recherche de documents avec BM25 (Langchain) ===")
            print("Tapez 'exit' pour quitter")

            while True:
                query = input("\nEntrez votre recherche: ")

                if query.lower() == 'exit':
                    print("Au revoir!")
                    break

                if query.strip():
                    try:
                        results = self.search_documents(query)

                        if results:
                            print(f"\n{len(results)} documents trouvés:")
                            for result in results:
                                print(f"\n[{result['rank']}] Score: {result.get('score', 'N/A')}")
                                print(f"Source: {os.path.basename(result['source'])}")
                                print(f"Extrait: {result['content']}")

                            with open("resultats_recherche_bm25_langchain.txt", "w", encoding="utf-8") as f:
                                for result in results:
                                    f.write(f"\n[{result['rank']}] Score: {result.get('score', 'N/A')}\n")
                                    f.write(f"Source: {os.path.basename(result['source'])}\n")
                                    f.write(f"Extrait: {result['content']}\n")

                            print("Les résultats ont été sauvegardés dans 'resultats_recherche_bm25_langchain.txt'.")
                        else:
                            print("Aucun document trouvé pour cette requête.")
                    except Exception as e:
                        self.logger.error(f"Erreur lors de la recherche: {str(e)}")
                        print(f"Erreur lors de la recherche: {str(e)}")
                else:
                    print("Veuillez entrer une requête valide.")

        except Exception as e:
            self.logger.error(f"Erreur: {str(e)}")
        self.logger.info("Fin de l'exécution du script principal")

if __name__ == "__main__":
    documents_dir = r"D:\bureau\BD&AI 1\ci2\S2\th_info\cour"
    bm25_search = LangchainBM25DocumentSearch(documents_dir)
    bm25_search.main()