import os
import sys
import subprocess
import logging

class WhisperTranscriber:
    """Classe pour gérer la transcription audio avec le modèle Whisper"""
    
    def __init__(self, model_size="tiny", language="fr"):
        """
        Initialise le transcripteur
        
        Args:
            model_size (str): Taille du modèle (tiny, base, small, medium, large)
            language (str): Code langue (fr, en, etc.)
        """
        # Configurer le logger
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.model_size = model_size
        self.language = language
        self.model = None
        
        # Vérifier les dépendances
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Vérifie et importe les dépendances nécessaires"""
        try:
            global whisper
            import whisper
            self.logger.info("Module whisper trouvé et importé avec succès")
        except ImportError:
            self.logger.error("Le module whisper n'a pas été trouvé.")
            print("ERREUR: Le module whisper n'a pas été trouvé.")
            print("Assurez-vous que openai-whisper est installé avec:")
            print("pip install openai-whisper")
            print("\nSi vous l'avez déjà installé, essayez ces solutions:")
            print("1. Vérifiez votre environnement virtuel")
            print("2. Réinstallez avec: pip uninstall whisper && pip uninstall openai-whisper && pip install openai-whisper")
            print("3. Assurez-vous que FFmpeg est installé sur votre système")
            sys.exit(1)
    
    def check_ffmpeg(self):
        """Vérifie si FFmpeg est installé et accessible
        
        Returns:
            bool: True si FFmpeg est installé, False sinon
        """
        self.logger.info("Début de la vérification de FFmpeg")
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.logger.info("FFmpeg est installé et accessible")
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error("FFmpeg n'est pas installé ou n'est pas dans votre PATH.")
            print("ERREUR: FFmpeg n'est pas installé ou n'est pas dans votre PATH.")
            print("Veuillez installer FFmpeg: https://ffmpeg.org/download.html")
            print("Sous Windows, vous pouvez utiliser: choco install ffmpeg")
            print("Ensuite, redémarrez votre terminal ou votre ordinateur")
            return False
        finally:
            self.logger.info("Fin de la vérification de FFmpeg")
    
    def load_model(self):
        """Charge le modèle Whisper
        
        Returns:
            bool: True si le modèle est chargé avec succès, False sinon
        """
        try:
            print(f"Chargement du modèle Whisper ({self.model_size})...")
            self.model = whisper.load_model(self.model_size)
            self.logger.info(f"Modèle {self.model_size} chargé avec succès")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            print(f"ERREUR lors du chargement du modèle: {str(e)}")
            return False
    
    def transcribe(self, file_path):
        """
        Transcrit un fichier audio en texte
        
        Args:
            file_path (str): Chemin vers le fichier audio
            
        Returns:
            str: Texte transcrit ou None en cas d'erreur
        """
        self.logger.info(f"Début de la transcription du fichier: {file_path}")
        
        try:
            # Vérifier que le fichier existe
            if not os.path.exists(file_path):
                self.logger.error(f"Le fichier {file_path} n'existe pas.")
                print(f"ERREUR: Le fichier {file_path} n'existe pas.")
                return None
            
            # Charger le modèle s'il n'est pas déjà chargé
            if self.model is None:
                if not self.load_model():
                    return None
            
            print(f"Transcription du fichier: {file_path}")
            result = self.model.transcribe(file_path, language=self.language)
            
            print("Transcription terminée!")
            self.logger.info("Transcription terminée avec succès")
            return result["text"]
        except Exception as e:
            self.logger.error(f"Erreur pendant la transcription: {str(e)}")
            print(f"ERREUR pendant la transcription: {str(e)}")
            return None
        finally:
            self.logger.info(f"Fin de la transcription du fichier: {file_path}")
    
    def save_transcription(self, text, file_path):
        """
        Enregistre la transcription dans un fichier texte
        
        Args:
            text (str): Texte à enregistrer
            file_path (str): Chemin du fichier audio original
            
        Returns:
            str: Chemin du fichier de sortie ou None en cas d'erreur
        """
        try:
            output_file = f"{os.path.splitext(file_path)[0]}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"Transcription enregistrée dans: {output_file}")
            self.logger.info(f"Transcription enregistrée dans: {output_file}")
            return output_file
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement de la transcription: {str(e)}")
            print(f"ERREUR lors de l'enregistrement de la transcription: {str(e)}")
            return None


class TranscriptionApp:
    """Classe principale de l'application de transcription"""
    
    def __init__(self):
        """Initialise l'application de transcription"""
        self.logger = logging.getLogger(__name__)
    
    def print_system_info(self, file_path, language, model_size):
        """Affiche les informations système
        
        Args:
            file_path (str): Chemin du fichier audio
            language (str): Code langue
            model_size (str): Taille du modèle
        """
        print("Informations système:")
        print(f"Python: {sys.version}")
        print(f"Chemin Python: {sys.executable}")
        print(f"Version whisper: {whisper.__version__ if hasattr(whisper, '__version__') else 'Non disponible'}")
        print(f"Fichier audio: {file_path}")
        print(f"Langue: {language}")
        print(f"Modèle: {model_size}")
    
    def display_usage(self):
        """Affiche le message d'aide pour l'utilisation de l'application"""
        print("\nUtilisation:")
        print("python transcription.py <fichier_audio> [langue] [taille_modèle]")
        print("\nExemples:")
        print("python transcription.py audio.mp3")
        print("python transcription.py audio.mp3 fr")
        print("python transcription.py audio.mp3 en medium")
        print("\nOptions:")
        print("  <fichier_audio>: Chemin vers le fichier audio à transcrire (obligatoire)")
        print("  [langue]: Code de langue (par défaut: fr)")
        print("  [taille_modèle]: Taille du modèle Whisper (tiny, base, small, medium, large) (par défaut: tiny)")
    
    def run(self, args=None):
        """
        Exécute l'application avec les arguments spécifiés
        
        Args:
            args (list): Liste d'arguments (par défaut utilise sys.argv)
            
        Returns:
            str: Texte transcrit ou None en cas d'erreur
        """
        self.logger.info("Début de l'application")
        
        # Utiliser les arguments fournis ou sys.argv
        if args is None:
            args = sys.argv
        
        # Vérifier si un fichier audio a été spécifié
        if len(args) < 2:
            self.logger.error("Aucun fichier audio spécifié.")
            print("ERREUR: Aucun fichier audio spécifié.")
            self.display_usage()
            return None
        
        # Récupérer le chemin du fichier audio
        audio_file = os.path.abspath(args[1])
        
        # Déterminer la langue (par défaut: français)
        language = "fr"
        if len(args) > 2:
            language = args[2]
        
        # Déterminer la taille du modèle (par défaut: tiny)
        model_size = "tiny"
        if len(args) > 3:
            model_size = args[3]
        
        # Créer le transcripteur
        transcriber = WhisperTranscriber(model_size, language)
        
        # Vérifier si FFmpeg est installé
        if not transcriber.check_ffmpeg():
            return None
        
        # Afficher les informations système
        self.print_system_info(audio_file, language, model_size)
        
        # Transcrire l'audio
        transcription = transcriber.transcribe(audio_file)
        
        if transcription:
            print("\nRésultat de la transcription:")
            print("============================")
            print(transcription)
            print("============================\n")
            
            # Enregistrer dans un fichier texte
            transcriber.save_transcription(transcription, audio_file)
            
            return transcription
        else:
            self.logger.error("La transcription a échoué.")
            print("La transcription a échoué.")
            return None

    def batch_transcribe(self, file_paths, language="fr", model_size="tiny"):
        """
        Transcrit plusieurs fichiers audio en lot
        
        Args:
            file_paths (list): Liste des chemins de fichiers audio
            language (str): Code langue (fr, en, etc.)
            model_size (str): Taille du modèle (tiny, base, small, medium, large)
            
        Returns:
            dict: Dictionnaire des transcriptions avec les chemins de fichiers comme clés
        """
        self.logger.info(f"Début de la transcription par lot de {len(file_paths)} fichiers")
        print(f"Transcription par lot de {len(file_paths)} fichiers...")
        
        # Créer le transcripteur
        transcriber = WhisperTranscriber(model_size, language)
        
        # Vérifier si FFmpeg est installé
        if not transcriber.check_ffmpeg():
            return None
        
        results = {}
        for i, file_path in enumerate(file_paths, 1):
            print(f"\nTraitement du fichier {i}/{len(file_paths)}: {file_path}")
            transcription = transcriber.transcribe(file_path)
            
            if transcription:
                # Enregistrer dans un fichier texte
                transcriber.save_transcription(transcription, file_path)
                results[file_path] = transcription
            else:
                print(f"Échec de la transcription pour {file_path}")
        
        print(f"\nTranscription par lot terminée. {len(results)}/{len(file_paths)} fichiers traités avec succès.")
        self.logger.info(f"Transcription par lot terminée. {len(results)}/{len(file_paths)} fichiers traités avec succès.")
        
        return results


def main():
    """Point d'entrée principal du programme"""
    app = TranscriptionApp()
    
    # Pour un seul fichier (comportement par défaut)
    # result = app.run()
    
    # Pour transcrire plusieurs fichiers
    files = ["test.mp3"]
    results = app.batch_transcribe(files, language="fr", model_size="medium")
    
    sys.exit(0 if results else 1)


if __name__ == "__main__":
    # Si le script est exécuté directement, appeler la fonction main
    main()