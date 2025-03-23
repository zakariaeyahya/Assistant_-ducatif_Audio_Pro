import os
import sys
import os

# Ajoutez le chemin du répertoire parent au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Imports pour Groq et LangChain
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.evaluation import load_evaluator, EvaluatorType

# Import MLflow
import mlflow
from mlflow.tracking import MlflowClient

# Import de notre recherche hybride
from services.HybridSearch import HybridSearch, SearchResult

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("educational_assistant")

class EducationalAssistant:
    def __init__(
        self,
        documents_dir: str,
        qdrant_path: str,
        collection_name: str = "asr_docs",
        history_file: str = "conversation_history.json",
        analytics_file: str = "analytics_data.json",
        api_key: str = "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",
        mlflow_tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "assistant_educatif"
    ):
        """
        Initialise l'assistant éducatif avec suivi MLflow.
        
        Args:
            documents_dir: Répertoire contenant les documents PDF
            qdrant_path: Chemin vers le stockage Qdrant
            collection_name: Nom de la collection Qdrant
            history_file: Fichier pour stocker l'historique des conversations
            analytics_file: Fichier pour stocker les données analytiques
            api_key: Clé API Groq
            mlflow_tracking_uri: URI du serveur MLflow
            experiment_name: Nom de l'expérience MLflow
        """
        self.api_key = api_key
        self.history_file = history_file
        self.analytics_file = analytics_file
        
        # Configuration MLflow
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name
        self._setup_mlflow()
        
        # Initialiser le modèle LangChain Groq
        self.llm = ChatGroq(
            api_key=self.api_key,
            model_name="llama3-70b-8192"
        )
        
        # Initialiser la recherche hybride
        self.hybrid_search = HybridSearch(
            documents_dir=documents_dir,
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            bm25_results_count=2,
            vector_results_count=2
        )
        
        # Système de stockage d'historique pour LangChain
        self.message_history_store = {}
        
        # Initialiser l'évaluateur
        try:
            self.evaluator = load_evaluator(
                EvaluatorType.LABELED_SCORE_STRING,  # Évaluateur avec score de 1 à 10
                llm=self.llm
            )
            logger.info("Évaluateur initialisé avec succès.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'évaluateur: {e}")
            self.evaluator = None
        
        # Définir le prompt système 
        self.system_prompt = """
        Tu es un assistant éducatif conçu pour aider les élèves à comprendre des concepts plutôt que de leur 
        donner directement les réponses. Ton objectif est de les guider vers la compréhension en:
        
        1. Posant des questions pour clarifier leur niveau de compréhension actuel
        2. Fournissant des indices et des explications qui les aident à raisonner par eux-mêmes
        3. Proposant des analogies pour faciliter la compréhension
        4. Encourageant la réflexion critique et l'auto-apprentissage
        5. Utilisant les documents de référence pour contextualiser tes explications
        
        N'offre JAMAIS la solution complète à un problème ou exercice. Oriente plutôt l'élève 
        vers la méthode de résolution appropriée. Si l'élève semble frustré, donne-lui un indice 
        plus direct mais jamais la réponse entière.
        
        Utilise les informations extraites des documents de référence pour enrichir tes explications,
        mais ne te contente pas de les répéter mot pour mot. Reformule les informations de manière
        pédagogique et adaptée au niveau de l'élève.
        """
        
        # Configurer le template de prompt
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
            ("system", "{context}")
        ])
        
        # Créer la chaîne avec gestion d'historique
        self.chain_with_history = self._create_chain_with_history()
        
        # Charger l'historique complet des conversations précédentes
        self.conversation_record = self._load_conversation_record()
        
        # Charger les données analytiques précédentes
        self.analytics_data = self._load_analytics_data()
        
        # Démarrer une session MLflow
        self.mlflow_run_id = self._start_mlflow_run()
        
        logger.info("Assistant éducatif initialisé avec succès.")
    
    def _setup_mlflow(self):
        """Configure MLflow pour le suivi des expériences."""
        try:
            # Configurer l'URI de suivi MLflow
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Créer ou obtenir l'expérience
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    tags={
                        "description": "Suivi des performances de l'assistant éducatif",
                        "version": "1.0.0"
                    }
                )
                logger.info(f"Expérience MLflow créée avec ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Utilisation de l'expérience MLflow existante avec ID: {experiment_id}")
                
            mlflow.set_experiment(self.experiment_name)
            logger.info("Configuration MLflow terminée avec succès.")
            
        except Exception as e:
            logger.error(f"Erreur lors de la configuration MLflow: {e}")
            raise
    
    def _start_mlflow_run(self):
        """Démarre une nouvelle exécution MLflow et retourne l'ID d'exécution."""
        try:
            current_run = mlflow.start_run(
                run_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={
                    "model": "llama3-70b-8192",
                    "api": "groq",
                    "search_method": "hybrid",
                }
            )
            
            # Loguer les paramètres du système
            mlflow.log_params({
                "llm_model": "llama3-70b-8192",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "bm25_results_count": 2,
                "vector_results_count": 2,
                "system_prompt_length": len(self.system_prompt)
            })
            
            logger.info(f"Session MLflow démarrée avec ID: {current_run.info.run_id}")
            return current_run.info.run_id
            
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de la session MLflow: {e}")
            return None
    
    def _log_interaction_to_mlflow(self, query_length: int, response_length: int, evaluation_results: Dict[str, Any], 
                                  search_result_count: int, bm25_count: int, vector_count: int):
        """Enregistre les métriques d'interaction dans MLflow."""
        if not self.mlflow_run_id:
            logger.warning("Impossible d'enregistrer les métriques car aucune session MLflow n'est active")
            return
            
        try:
            metrics = {
                "query_length": query_length,
                "response_length": response_length,
                "total_search_results": search_result_count,
                "bm25_results": bm25_count,
                "vector_results": vector_count,
                "response_time_ms": 0  # À mettre à jour avec le temps réel
            }
            
            # Ajouter les scores d'évaluation
            if isinstance(evaluation_results, dict):
                if "global_score" in evaluation_results:
                    metrics["evaluation_score"] = evaluation_results["global_score"]
                
                # Ajouter les scores par critère
                for key, value in evaluation_results.items():
                    if key != "global_score" and isinstance(value, dict) and "score" in value:
                        metrics[f"score_{key.lower()}"] = value["score"]
            
            # Enregistrer les métriques dans MLflow
            mlflow.log_metrics(metrics)
            logger.info("Métriques enregistrées dans MLflow avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement des métriques dans MLflow: {e}")
    
    def _log_artifact_to_mlflow(self, name: str, path: str):
        """Enregistre un artefact dans MLflow."""
        if not self.mlflow_run_id:
            return
            
        try:
            if os.path.exists(path):
                mlflow.log_artifact(path, name)
                logger.info(f"Artefact '{name}' enregistré dans MLflow avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'artefact '{name}' dans MLflow: {e}")
    
    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Récupère ou crée un historique de session."""
        if session_id not in self.message_history_store:
            self.message_history_store[session_id] = InMemoryChatMessageHistory()
        return self.message_history_store[session_id]
    
    def _create_chain_with_history(self):
        """Crée une chaîne LangChain avec gestion d'historique."""
        chain = self.prompt_template | self.llm
        
        # Ajouter la gestion d'historique
        return RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
    
    def _load_conversation_record(self) -> List[Dict[str, Any]]:
        """Charge l'historique complet des conversations depuis un fichier JSON."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'historique: {e}")
                return []
        return []
    
    def _save_conversation_record(self):
        """Sauvegarde l'historique complet des conversations dans un fichier JSON."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_record, f, ensure_ascii=False, indent=4)
            logger.info(f"Historique de conversation sauvegardé dans {self.history_file}")
            
            # Enregistrer également l'historique comme artefact MLflow
            self._log_artifact_to_mlflow("conversation_history", self.history_file)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {e}")
    
    def _load_analytics_data(self) -> List[Dict[str, Any]]:
        """Charge les données analytiques depuis un fichier JSON."""
        if os.path.exists(self.analytics_file):
            try:
                with open(self.analytics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données analytiques: {e}")
                return []
        return []
    
    def _save_analytics_data(self):
        """Sauvegarde les données analytiques dans un fichier JSON."""
        try:
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(self.analytics_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Données analytiques sauvegardées dans {self.analytics_file}")
            
            # Enregistrer également les données analytiques comme artefact MLflow
            self._log_artifact_to_mlflow("analytics_data", self.analytics_file)
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données analytiques: {e}")
    
    def _search_documents(self, query: str) -> List[SearchResult]:
        """Recherche des documents pertinents pour la requête."""
        try:
            results = self.hybrid_search.search(query)
            return results
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de documents: {e}")
            return []
    
    def _format_search_results(self, results: List[SearchResult]) -> str:
        """Formate les résultats de recherche pour le contexte."""
        if not results:
            return "Aucune information pertinente trouvée dans les documents."
        
        formatted_results = "Informations pertinentes trouvées dans les documents:\n\n"
        for i, result in enumerate(results):
            formatted_results += f"[Document {i+1} - {result.method.upper()} (score: {result.score:.2f})]\n"
            formatted_results += f"{result.text[:500]}...\n\n"
        
        return formatted_results
    
    def _find_similar_past_queries(self, query: str, max_similar: int = 2) -> List[Dict[str, Any]]:
        """
        Trouve des requêtes similaires dans l'historique des conversations.
        Utilise une simple correspondance de mots-clés pour l'exemple.
        """
        similar_queries = []
        query_words = set([word.lower() for word in query.split() if len(word) > 3])
        
        if not query_words:
            return []
        
        for entry in reversed(self.conversation_record):
            past_query = entry.get("question", "").lower()
            past_query_words = set([word.lower() for word in past_query.split() if len(word) > 3])
            
            # Calculer un score de similarité simple (intersection des mots-clés)
            if past_query_words and query_words:
                similarity = len(query_words.intersection(past_query_words)) / len(query_words.union(past_query_words))
                
                if similarity > 0.2:  # Seuil de similarité
                    similar_queries.append({
                        "question": entry.get("question", ""),
                        "answer": entry.get("answer", ""),
                        "similarity": similarity
                    })
                    
                    if len(similar_queries) >= max_similar:
                        break
        
        return similar_queries
    
    def _evaluate_response(self, query: str, response: str, context: str) -> Dict[str, Any]:
        """
        Évalue la qualité de la réponse générée.
        
        Args:
            query: Question de l'utilisateur
            response: Réponse générée par l'IA
            context: Contexte fourni pour la génération
            
        Returns:
            Dictionnaire contenant les scores d'évaluation
        """
        if not self.evaluator:
            return {"error": "Évaluateur non disponible"}
        
        try:
            # Critères d'évaluation
            criteria = [
                "Pertinence: La réponse est-elle pertinente par rapport à la question?",
                "Guidage pédagogique: La réponse guide-t-elle l'élève sans donner la solution directement?",
                "Utilisation du contexte: La réponse utilise-t-elle bien les informations fournies?",
                "Clarté: La réponse est-elle claire et facile à comprendre?"
            ]
            
            evaluation_results = {}
            
            # Évaluer chaque critère
            for criterion in criteria:
                try:
                    eval_result = self.evaluator.evaluate_strings(
                        prediction=response,
                        reference=query,
                        input=f"Critère: {criterion}\nContexte: {context[:500]}..."
                    )
                    
                    criterion_name = criterion.split(":")[0]
                    evaluation_results[criterion_name] = {
                        "score": eval_result.get("score", 0),
                        "reasoning": eval_result.get("reasoning", "")
                    }
                    
                except Exception as e:
                    logger.error(f"Erreur lors de l'évaluation du critère '{criterion}': {e}")
                    evaluation_results[criterion.split(":")[0]] = {"error": str(e)}
            
            # Calculer un score global
            scores = [v.get("score", 0) for k, v in evaluation_results.items() if isinstance(v, dict) and "score" in v]
            global_score = sum(scores) / len(scores) if scores else 0
            
            evaluation_results["global_score"] = global_score
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation: {e}")
            return {"error": str(e)}
    
    def store_interaction_data(self, 
                               query: str, 
                               response: str, 
                               session_id: str,
                               search_results: List[SearchResult],
                               evaluation_results: Dict[str, Any]):
        """
        Stocke les données d'interaction pour analyse ultérieure.
        
        Args:
            query: Question de l'utilisateur
            response: Réponse générée
            session_id: ID de la session
            search_results: Résultats de la recherche
            evaluation_results: Résultats de l'évaluation
        """
        # Structurer les données d'analytics
        analytics_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": query,
            "response": response,
            "search_results": [
                {
                    "method": result.method,
                    "score": result.score,
                    "source": result.source,
                    "text_excerpt": result.text[:300] + "..." if len(result.text) > 300 else result.text
                }
                for result in search_results
            ],
            "evaluation": evaluation_results
        }
        
        # Ajouter aux données analytiques
        self.analytics_data.append(analytics_entry)
        
        # Sauvegarder les données
        self._save_analytics_data()
        
        logger.info(f"Données d'interaction stockées pour la requête: '{query[:50]}...'")
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Traite une requête et génère une réponse guidée.
        
        Args:
            query: La question ou requête de l'élève
            session_id: Identifiant de session pour l'historique de conversation
            
        Returns:
            Réponse guidée de l'assistant
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        start_time = datetime.now()
        logger.info(f"Traitement de la requête pour session {session_id}: '{query}'")
        
        # Rechercher des documents pertinents
        search_results = self._search_documents(query)
        context = self._format_search_results(search_results)
        
        # Compter les résultats par méthode
        bm25_count = sum(1 for r in search_results if r.method.lower() == "bm25")
        vector_count = sum(1 for r in search_results if r.method.lower() == "vector")
        
        # Trouver des requêtes similaires dans l'historique
        similar_queries = self._find_similar_past_queries(query)
        
        # Ajouter les informations de requêtes similaires au contexte
        if similar_queries:
            context += "\n\nQuestions similaires posées précédemment:\n\n"
            for i, entry in enumerate(similar_queries):
                context += f"Question: {entry['question']}\n"
                context += f"Réponse précédente: {entry['answer']}\n\n"
        
        # Ajouter des instructions finales au contexte
        context += "\nRappel: Guide l'élève vers la compréhension sans donner directement la réponse. Utilise les informations ci-dessus pour enrichir ton explication."
        
        try:
            # Invoquer la chaîne avec l'historique
            response = self.chain_with_history.invoke(
                {"input": query, "context": context},
                config={"configurable": {"session_id": session_id}}
            )
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Calculer le temps de réponse
            response_time = (datetime.now() - start_time).total_seconds() * 1000  # en millisecondes
            
            # Évaluer la réponse
            evaluation_results = self._evaluate_response(query, answer, context)
            
            # Log des métriques dans MLflow
            self._log_interaction_to_mlflow(
                query_length=len(query),
                response_length=len(answer),
                evaluation_results=evaluation_results,
                search_result_count=len(search_results),
                bm25_count=bm25_count,
                vector_count=vector_count
            )
            
            # Stocker les données d'interaction
            self.store_interaction_data(
                query=query,
                response=answer,
                session_id=session_id,
                search_results=search_results,
                evaluation_results=evaluation_results
            )
            
            # Enregistrer cette conversation dans l'historique complet
            self.conversation_record.append({
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "question": query,
                "answer": answer,
                "evaluation": evaluation_results
            })
            
            # Sauvegarder l'historique mis à jour
            self._save_conversation_record()
            
            logger.info(f"Réponse générée avec succès pour session {session_id}")
            
            # Afficher le score d'évaluation
            global_score = evaluation_results.get("global_score", 0)
            if global_score:
                logger.info(f"Score d'évaluation de la réponse: {global_score:.2f}/10")
            
            return answer
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {e}")
            
            # Log l'erreur dans MLflow
            if self.mlflow_run_id:
                mlflow.log_param("error", str(e))
                
            return f"Désolé, je n'ai pas pu générer une réponse. Erreur: {str(e)}"
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des données analytiques stockées.
        
        Returns:
            Dictionnaire contenant des statistiques sur les interactions
        """
        if not self.analytics_data:
            return {"error": "Aucune donnée analytique disponible"}
        
        try:
            total_interactions = len(self.analytics_data)
            
            # Extraire les scores globaux
            global_scores = [
                entry.get("evaluation", {}).get("global_score", 0) 
                for entry in self.analytics_data 
                if isinstance(entry.get("evaluation"), dict) and "global_score" in entry.get("evaluation", {})
            ]
            
            # Calculer les statistiques des scores
            avg_score = sum(global_scores) / len(global_scores) if global_scores else 0
            max_score = max(global_scores) if global_scores else 0
            min_score = min(global_scores) if global_scores else 0
            
            # Compter les occurrences de sources de documents
            sources = []
            for entry in self.analytics_data:
                for result in entry.get("search_results", []):
                    sources.append(result.get("source", ""))
            
            source_counts = {}
            for source in sources:
                source_name = os.path.basename(source) if source else "Unknown"
                if source_name in source_counts:
                    source_counts[source_name] += 1
                else:
                    source_counts[source_name] = 1
            
            # Compter les méthodes de recherche utilisées
            methods = []
            for entry in self.analytics_data:
                for result in entry.get("search_results", []):
                    methods.append(result.get("method", ""))
            
            method_counts = {}
            for method in methods:
                if method in method_counts:
                    method_counts[method] += 1
                else:
                    method_counts[method] = 1
            
            summary = {
                "total_interactions": total_interactions,
                "evaluation_scores": {
                    "average": avg_score,
                    "max": max_score,
                    "min": min_score
                },
                "source_usage": source_counts,
                "search_methods": method_counts
            }
            
            # Log le résumé dans MLflow
            if self.mlflow_run_id:
                mlflow.log_metrics({
                    "total_interactions": total_interactions,
                    "avg_evaluation_score": avg_score,
                    "max_evaluation_score": max_score,
                    "min_evaluation_score": min_score
                })
                
                # Créer un fichier JSON de résumé et l'enregistrer comme artefact
                summary_file = "analytics_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, ensure_ascii=False, indent=4)
                
                self._log_artifact_to_mlflow("summary", summary_file)
            
            return summary
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du résumé analytique: {e}")
            return {"error": str(e)}
    
    def finish_session(self):
        """Termine la session MLflow et effectue les opérations de nettoyage."""
        try:
            # Générer et enregistrer le résumé final
            summary = self.get_analytics_summary()
            
            # Terminer la session MLflow
            if self.mlflow_run_id:
                mlflow.end_run()
                logger.info(f"Session MLflow {self.mlflow_run_id} terminée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de la finalisation de la session: {e}")
    
    def run_interactive(self):
        """Lance une interface interactive en ligne de commande."""
        print("\n=== Assistant Éducatif Interactif avec suivi MLflow ===")
        print("(Tapez 'exit' pour quitter, 'new' pour démarrer une nouvelle session, 'stats' pour voir les statistiques)")
        
        session_id = str(uuid.uuid4())
        print(f"Session démarrée: {session_id}")
        
        try:
            while True:
                query = input("\nVotre question : ")
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Au revoir!")
                    break
                
                if query.lower() == 'new':
                    session_id = str(uuid.uuid4())
                    print(f"Nouvelle session démarrée: {session_id}")
                    continue
                
                if query.lower() == 'stats':
                    stats = self.get_analytics_summary()
                    print("\n=== Statistiques d'utilisation ===")
                    print(f"Total des interactions: {stats.get('total_interactions', 0)}")
                    print(f"Score moyen des réponses: {stats.get('evaluation_scores', {}).get('average', 0):.2f}/10")
                    print("\nMéthodes de recherche utilisées:")
                    for method, count in stats.get('search_methods', {}).items():
                        print(f"  - {method}: {count}")
                    print("\nSources les plus utilisées:")
                    for source, count in sorted(stats.get('source_usage', {}).items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"  - {source}: {count}")
                    continue
                
                if query.strip():
                    response = self.process_query(query, session_id)
                    print("\nAssistant:")
                    print(response)
                else:
                    print("Veuillez entrer une question valide.")
        finally:
            # S'assurer de terminer proprement la session MLflow
            self.finish_session()

if __name__ == "__main__":
    # Configuration
    DOCUMENTS_DIR = "D:/bureau/BD&AI 1/ci2/S2/th_info/cour"
    QDRANT_PATH = "D:/bureau/BD&AI 1/ci2/S2/tec_veille/mini_projet/local_qdrant_storage"
    COLLECTION_NAME = "asr_docs"
    
    # Configuration MLflow
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    EXPERIMENT_NAME = "assistant_educatif"
    
    # Créer et exécuter l'assistant
    assistant = EducationalAssistant(
        documents_dir=DOCUMENTS_DIR,
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=EXPERIMENT_NAME
    )
    
    assistant.run_interactive()
