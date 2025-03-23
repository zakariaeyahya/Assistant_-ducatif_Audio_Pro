import sys
import os

# Ajoutez le chemin du répertoire parent au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import logging
from typing import List, Dict, Any
import asyncio
from pydantic import BaseModel

# Import depuis votre nouveau fichier
from Searching.LangchainBM25DocumentSearch import LangchainBM25DocumentSearch
from Searching.vector_search import QdrantDocumentSearch

# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("hybrid_search")

class SearchResult(BaseModel):
    text: str
    source: str
    score: float
    method: str  # "vector" ou "bm25"

class HybridSearch:
    def __init__(
        self,
        documents_dir: str,
        qdrant_path: str,
        collection_name: str = "asr_docs",
        bm25_results_count: int = 2,    # Nombre de résultats BM25 à retourner
        vector_results_count: int = 2,  # Nombre de résultats vectoriels à retourner
    ):
        """
        Initialise la recherche hybride qui combine BM25 et la recherche vectorielle.
        
        Args:
            documents_dir: Répertoire contenant les documents PDF
            qdrant_path: Chemin vers le stockage Qdrant
            collection_name: Nom de la collection Qdrant
            bm25_results_count: Nombre de résultats à retourner depuis la recherche BM25
            vector_results_count: Nombre de résultats à retourner depuis la recherche vectorielle
        """
        self.bm25_results_count = bm25_results_count
        self.vector_results_count = vector_results_count
        
        # Initialisation de la recherche BM25 avec Langchain
        self.bm25_search = LangchainBM25DocumentSearch(
            documents_dir=documents_dir,
            top_k=bm25_results_count
        )
        
        # Initialisation de la recherche vectorielle
        self.vector_search = QdrantDocumentSearch(
            qdrant_path=qdrant_path,
            collection_name=collection_name
        )
        
        # Préparation des ressources BM25
        chunks = self.bm25_search.get_document_chunks()
        if chunks:
            self.bm25_retriever = self.bm25_search.initialize_bm25_retriever(chunks)
            logger.info("Recherche BM25 initialisée avec succès.")
        else:
            logger.warning("Aucun document trouvé pour l'initialisation de la recherche BM25.")
        
        logger.info(f"Configuration de la recherche hybride terminée avec {bm25_results_count} résultats BM25 et {vector_results_count} résultats vectoriels.")

    def search(self, query: str) -> List[SearchResult]:
        """
        Effectue les deux méthodes de recherche et combine les résultats.
        
        Args:
            query: La requête de recherche
            
        Returns:
            Liste des résultats de recherche combinés
        """
        logger.info(f"Exécution d'une recherche hybride pour la requête: '{query}'")
        
        # Obtenir les résultats BM25
        try:
            bm25_results = self.bm25_search.search_documents(query)
            formatted_bm25_results = []
            
            for result in bm25_results[:self.bm25_results_count]:
                formatted_bm25_results.append(SearchResult(
                    text=result["content"],
                    source=result["source"],
                    score=result.get("score", 0.0) if isinstance(result.get("score"), (int, float)) else 0.0,
                    method="bm25"
                ))
            
            logger.info(f"La recherche BM25 a retourné {len(formatted_bm25_results)} résultats.")
        except Exception as e:
            logger.error(f"Erreur dans la recherche BM25: {str(e)}")
            formatted_bm25_results = []
        
        # Obtenir les résultats vectoriels
        try:
            vector_results = self.vector_search.search(query, k=self.vector_results_count)
            formatted_vector_results = []
            
            for i, text in enumerate(vector_results):
                formatted_vector_results.append(SearchResult(
                    text=text[:2000],  # Limiter la longueur du texte
                    source="Recherche Vectorielle",
                    score=1.0 - (i * 0.1),  # Score décroissant simple pour l'ordre
                    method="vector"
                ))
            
            logger.info(f"La recherche vectorielle a retourné {len(formatted_vector_results)} résultats.")
        except Exception as e:
            logger.error(f"Erreur dans la recherche vectorielle: {str(e)}")
            formatted_vector_results = []
        
        # Combiner les résultats
        combined_results = formatted_bm25_results + formatted_vector_results
        
        # Trier par score (le plus élevé en premier)
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"La recherche hybride a retourné {len(combined_results)} résultats.")
        return combined_results

    def save_results(self, results: List[SearchResult], output_file: str = "resultats_recherche_hybride.txt"):
        """Enregistrer les résultats de recherche dans un fichier."""
        logger.info(f"Enregistrement de {len(results)} résultats de recherche dans {output_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Résultats de recherche hybride\n")
            f.write(f"===========================\n\n")
            
            for i, result in enumerate(results):
                f.write(f"--- Résultat {i+1} ---\n")
                f.write(f"Méthode: {result.method}\n")
                f.write(f"Score: {result.score:.4f}\n")
                f.write(f"Source: {result.source}\n")
                f.write(f"Texte: {result.text[:2000]}...\n\n")  # Limiter à 2000 caractères
        
        logger.info(f"Résultats enregistrés dans {output_file}")

    def display_results_by_method(self, results: List[SearchResult]):
        """
        Affiche les résultats regroupés par méthode de recherche.
        """
        # Séparer les résultats par méthode
        bm25_results = [r for r in results if r.method == "bm25"]
        vector_results = [r for r in results if r.method == "vector"]
        
        # Afficher les résultats BM25
        print("\n--- Résultats BM25 ---")
        if bm25_results:
            for i, res in enumerate(bm25_results):
                print(f"\nRésultat BM25 #{i+1}:")
                print(f"Score: {res.score:.4f}")
                print(f"Source: {os.path.basename(res.source)}")
                print(f"Texte: {res.text[:500]}...")  
        else:
            print("Aucun résultat BM25 trouvé.")
        
        # Afficher les résultats vectoriels
        print("\n--- Résultats de recherche vectorielle ---")
        if vector_results:
            for i, res in enumerate(vector_results):
                print(f"\nRésultat vectoriel #{i+1}:")
                print(f"Score: {res.score:.4f}")
                print(f"Source: {res.source}")
                print(f"Texte: {res.text[:500]}...")   
        else:
            print("Aucun résultat vectoriel trouvé.")

def main():
    # Configuration
    DOCUMENTS_DIR = "D:/bureau/BD&AI 1/ci2/S2/th_info/cour"
    QDRANT_PATH = "D:/bureau/BD&AI 1/ci2/S2/tec_veille/mini_projet/local_qdrant_storage"
    COLLECTION_NAME = "asr_docs"
    
    # Initialiser la recherche hybride
    hybrid_search = HybridSearch(
        documents_dir=DOCUMENTS_DIR,
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        bm25_results_count=2,    # Exactement 2 résultats pour BM25
        vector_results_count=2   # Exactement 2 résultats pour la recherche vectorielle
    )
    
    # Exécuter la recherche interactive
    print("\n=== Recherche hybride de documents ===")
    print("Tapez 'exit' pour quitter.")
    
    while True:
        query = input("\nEntrez votre requête : ")
        
        if query.lower() == 'exit':
            print("Au revoir !")
            break
        
        if query.strip():
            try:
                # Effectuer la recherche hybride
                results = hybrid_search.search(query)
                
                if results:
                    print(f"\n{len(results)} résultats trouvés au total :")
                    
                    # Afficher les résultats regroupés par méthode
                    hybrid_search.display_results_by_method(results)
                    
                    # Afficher tous les résultats ensemble
                    print("\n--- Tous les résultats combinés ---")
                    for i, res in enumerate(results):
                        print(f"\nRésultat {i+1} ({res.method}):")
                        print(f"Score: {res.score:.4f}")
                        print(f"Source: {os.path.basename(res.source) if res.method == 'bm25' else res.source}")
                        print(f"Texte: {res.text[:500]}...")  # Afficher les 500 premiers caractères
                    
                    # Enregistrer les résultats dans un fichier
                    hybrid_search.save_results(results)
                    print("\nLes résultats ont été sauvegardés dans 'resultats_recherche_hybride.txt'.")
                else:
                    print("Aucun résultat trouvé pour cette requête.")
            except Exception as e:
                print(f"Erreur lors de la recherche : {e}")
                logger.error(f"Erreur lors de la recherche : {str(e)}", exc_info=True)
        else:
            print("Veuillez entrer une requête valide.")

if __name__ == "__main__":
    main()