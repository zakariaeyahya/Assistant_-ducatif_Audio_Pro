# Projet d'Assistant Éducatif 

Ce projet combine plusieurs scripts Python pour créer un système d'assistant éducatif interactif, utilisant des techniques avancées de traitement du langage naturel (NLP) et de recherche hybride. Il inclut des fonctionnalités pour l'indexation de documents PDF, la recherche hybride, la transcription audio, et une interface utilisateur interactive.

## Table des Matières

1. [Introduction](#introduction)
2. [Fonctionnalités](#fonctionnalités)
3. [Configuration](#configuration)
4. [Utilisation](#utilisation)
5. [Structure du Projet](#structure-du-projet)
6. [Licence](#licence)

## Introduction

Ce projet vise à fournir un assistant éducatif capable de guider les étudiants dans leur apprentissage en utilisant des techniques de recherche avancées et un modèle de langage. Il utilise MLflow pour suivre les performances et les interactions, permettant ainsi une amélioration continue du système.

## Fonctionnalités

- **Indexation de Documents PDF** : Extrait et indexe le texte des fichiers PDF pour une recherche efficace.
- **Recherche Hybride** : Combine les méthodes de recherche BM25 et vectorielle pour des résultats plus pertinents.
- **Transcription Audio** : Transcrit des fichiers audio en texte en utilisant le modèle Whisper.
- **Assistant Éducatif** : Fournit des réponses guidées aux questions des étudiants.
- **Interface Utilisateur** : Application web interactive utilisant Streamlit pour interagir avec l'assistant.

## Configuration

### Prérequis

- Python 3.x
- Bibliothèques Python : `PyPDF2`, `LlamaIndex`, `Qdrant`, `Langchain`, `FastAPI`, `Streamlit`, `Whisper`, `MLflow`, etc.
- FFmpeg (pour la transcription audio)

### Installation

1. Clonez le dépôt :
   ```bash
   git clone <URL_DU_DEPOT>
   cd <NOM_DU_DEPOT>
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Assurez-vous que FFmpeg est installé et accessible dans votre PATH.

## Utilisation

### Indexation de Documents PDF

Pour indexer des documents PDF, exécutez :
```bash
python stockage/stock.py
```

### Recherche Hybride

Pour effectuer une recherche hybride, exécutez :
```bash
python services/HybridSearch.py
```

### Transcription Audio

Pour transcrire des fichiers audio, exécutez :
```bash
python services/audio_transcribe.py --file <CHEMIN_DU_FICHIER_AUDIO>
```

### Assistant Éducatif

Pour lancer l'assistant éducatif, exécutez :
```bash
python services/chat.py
```

### Interface Utilisateur

Pour démarrer l'application Streamlit, exécutez :
```bash
streamlit run frontend/streamlit_app.py
```

## Structure du Projet

- `stockage/` : Scripts pour l'indexation de documents PDF.
- `services/` : Scripts pour la recherche hybride, la transcription audio, et l'assistant éducatif.
- `Searching/` : Scripts pour la recherche vectorielle et BM25.
- `api/` : API FastAPI pour interagir avec l'assistant éducatif et le service de transcription audio.
- `frontend/` : Application Streamlit pour l'interface utilisateur.

## Licence

Ce projet est sous licence [Nom de la Licence]. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

Ce fichier README fournit une vue d'ensemble complète du projet, y compris les instructions pour la configuration et l'utilisation, ainsi que des informations sur la structure du projet et la manière de contribuer.
