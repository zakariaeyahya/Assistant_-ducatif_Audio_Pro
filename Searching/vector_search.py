import qdrant_client
from qdrant_client.http import models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict, Any, Optional

class QdrantDocumentSearch:
    def __init__(
        self, 
        qdrant_path: str, 
        collection_name: str, 
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        vector_size: int = 768,  # Taille des vecteurs pour le modèle all-mpnet-base-v2
        recreate_collection: bool = False
    ):
        """
        Initialise la classe pour effectuer des recherches dans une collection Qdrant 
        avec l'algorithme HNSW et la similarité cosinus.
        
        Args:
            qdrant_path (str): Chemin vers le stockage local de Qdrant.
            collection_name (str): Nom de la collection Qdrant.
            embedding_model (str): Modèle HuggingFace pour générer les embeddings des requêtes.
            vector_size (int): Dimension des vecteurs d'embedding.
            recreate_collection (bool): Si True, supprime et recrée la collection (utile pour les tests).
        """
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        self.client = qdrant_client.QdrantClient(path=self.qdrant_path)
        self.vector_size = vector_size
        
        # Vérifier si la collection existe, sinon la créer
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if recreate_collection and self.collection_name in collection_names:
            self.client.delete_collection(collection_name=self.collection_name)
            collection_names.remove(self.collection_name)
        
        if self.collection_name not in collection_names:
            self._create_collection_with_hnsw()
    
    def _create_collection_with_hnsw(self):
        """
        Crée une nouvelle collection avec l'index HNSW pour la recherche rapide.
        """
        # Configuration de l'index HNSW
        hnsw_config = models.HnswConfigDiff(
            m=16,                # Nombre de connexions par nœud (défaut: 16)
            ef_construct=100,    # Qualité de construction de l'index (défaut: 100)
            full_scan_threshold=10000  # Seuil pour le scan complet (défaut: 10000)
        )
        
        # Création de la collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,  # Utilisation de la similarité cosinus
                hnsw_config=hnsw_config,          # Configuration HNSW
            ),
        )
        print(f"Collection '{self.collection_name}' créée avec l'index HNSW et la similarité cosinus.")
    
    def search(self, query: str, k: int = 2, score_threshold: Optional[float] = None) -> List[str]:
        """
        Recherche des documents similaires à la requête en utilisant HNSW avec la similarité cosinus.
        
        Args:
            query (str): La requête de recherche.
            k (int): Le nombre de résultats à retourner.
            score_threshold (float, optional): Seuil minimal de similarité (entre 0 et 1).
        
        Returns:
            List[str]: Une liste des textes des documents les plus similaires.
        """
        try:
            # Génère l'embedding de la requête
            query_embedding = self.embed_model.get_text_embedding(query)
            
            # Paramètres de recherche pour HNSW
            search_params = models.SearchParams(
                hnsw_ef=128,  # Paramètre de qualité pour la recherche HNSW
                exact=False   # False pour utiliser HNSW
            )
            
            # Utiliser query_points au lieu de search (qui est déprécié)
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                query_filter=None,  # Pas de filtrage supplémentaire
                with_payload=True,  # Récupérer les données associées
                with_vectors=False,  # Pas besoin des vecteurs
                search_params=search_params,
                score_threshold=score_threshold
            )
            
            # Extrait les textes des résultats
            if hasattr(search_result, 'points') and search_result.points:
                return [point.payload.get("text", "") for point in search_result.points]
            else:
                print("La requête n'a retourné aucun résultat.")
                return []
                
        except Exception as e:
            print(f"Erreur lors de la recherche: {str(e)}")
            # En cas d'erreur avec l'API, essayons la méthode legacy (utile si la version de qdrant est ancienne)
            try:
                # Version alternative utilisant l'ancienne méthode search
                query_embedding = self.embed_model.get_text_embedding(query)
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=k
                )
                return [hit.payload.get("text", "") for hit in results]
            except Exception as e2:
                print(f"Erreur également avec la méthode legacy: {str(e2)}")
                return []
    
    def save_results(self, results: List[str], output_file: str = "resultats_recherche.txt"):
        """
        Sauvegarde les résultats de la recherche dans un fichier.
        
        Args:
            results (List[str]): Liste des textes des résultats.
            output_file (str): Nom du fichier de sortie. Par défaut, "resultats_recherche.txt".
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for i, res in enumerate(results):
                f.write(f"\nRésultat {i+1}:\n{res[:1000]}...\n")  # Écrit les 1000 premiers caractères
        print(f"Les résultats ont été sauvegardés dans '{output_file}'.")
    
    def run(self):
        """
        Exécute une interface de recherche interactive.
        """
        print("\n=== Recherche de documents avec Qdrant (HNSW + Cosinus) ===")
        print("Tapez 'exit' pour quitter.")
        
        while True:
            query = input("\nEntrez votre requête : ")
            
            if query.lower() == 'exit':
                print("Au revoir !")
                break
            
            if query.strip():
                results = self.search(query)
                if results:
                    print(f"\n{len(results)} résultats trouvés :")
                    for i, res in enumerate(results):
                        print(f"\nRésultat {i+1}:\n{res[:1000]}...")  # Affiche les 1000 premiers caractères
                    self.save_results(results)
                else:
                    print("Aucun résultat trouvé pour cette requête.")
            else:
                print("Veuillez entrer une requête valide.")


if __name__ == "__main__":
    # Configuration
    QDRANT_PATH = r"D:\bureau\BD&AI 1\ci2\S2\tec_veille\mini_projet\local_qdrant_storage"
    COLLECTION_NAME = "asr_docs"
    
    # Initialisation et recherche
    searcher = QdrantDocumentSearch(
        qdrant_path=QDRANT_PATH, 
        collection_name=COLLECTION_NAME
    )
    searcher.run()
