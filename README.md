
---

# Educational Assistant Project

This project combines several Python scripts to create an interactive educational assistant system, using advanced natural language processing (NLP) and hybrid search techniques. It includes features for PDF document indexing, hybrid search, audio transcription, and an interactive user interface.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [License](#license)

## Introduction
This project aims to provide an educational assistant capable of guiding students in their learning using advanced search techniques and a language model. It uses MLflow to track performance and interactions, enabling continuous system improvement.

## Features
- **PDF Document Indexing**: Extracts and indexes text from PDF files for efficient search.
- **Hybrid Search**: Combines BM25 and vector search methods for more relevant results.
- **Audio Transcription**: Transcribes audio files to text using the Whisper model.
- **Educational Assistant**: Provides guided answers to student questions.
- **User Interface**: Interactive web application using Streamlit to interact with the assistant.

## Setup
### Prerequisites
- Python 3.x
- Python Libraries: `PyPDF2`, `LlamaIndex`, `Qdrant`, `Langchain`, `FastAPI`, `Streamlit`, `Whisper`, `MLflow`, etc.
- FFmpeg (for audio transcription)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/zakariaeyahya/Assistant_-ducatif_Audio_Pro.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure FFmpeg is installed and available in your PATH.

## Usage
![image](https://github.com/user-attachments/assets/e5fa9758-f23d-410f-be42-79cd22d9e47d)

### PDF Document Indexing
To index PDF documents, run:
```bash
python stockage/stock.py
```

### Hybrid Search
To perform a hybrid search, run:
```bash
python services/HybridSearch.py
```

### Audio Transcription
To transcribe audio files, run:
```bash
python services/audio_transcribe.py --file <AUDIO_FILE_PATH>
```

### Educational Assistant
To launch the educational assistant, run:
```bash
python services/chat.py
```

### User Interface
To start the Streamlit application, run:
```bash
streamlit run frontend/streamlit_app.py
```

## Project Structure
- `stockage/`: Scripts for PDF document indexing.
- `services/`: Scripts for hybrid search, audio transcription, and the educational assistant.
- `Searching/`: Scripts for vector and BM25 search.
- `api/`: FastAPI for interacting with the educational assistant and audio transcription service.
- `frontend/`: Streamlit application for the user interface.

## License
This project is licensed under [License Name]. See the [LICENSE](LICENSE) file for more details.

---
