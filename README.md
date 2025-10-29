# AI-Resume-Matcher📄🔍

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Menna-Khalid/AI-Resume-Matcher)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers/index)

## 🌟 Project Overview
The **AI-Resume-Matcher** is a sophisticated, end-to-end solution designed to revolutionize the recruitment process by leveraging advanced Natural Language Processing (NLP) and Machine Learning techniques. This project provides a robust API and a potential web interface to accurately match candidate resumes (CVs) against specific job descriptions (JDs), offering a quantitative and objective measure of fit.

The core functionality is built around a pre-trained **Transformer-based model** and a **TF-IDF Vectorizer** combined with **Cosine Similarity** to calculate the semantic similarity between the two documents. This approach moves beyond simple keyword matching to understand the context and meaning of the text, ensuring a highly relevant matching score.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ba879a3b-e619-48d0-91f5-b5c2750e1d51" width="300">
  <img src="https://github.com/user-attachments/assets/a19641c4-b602-4ebf-a9c1-bd1b837e6020" width="300">
  <img src="https://github.com/user-attachments/assets/d4e630ec-9548-49de-9446-3c661b85e9d4" width="300">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/69819dc8-d43a-495f-84e7-7fc6d863ec18" width="300">
  <img src="https://github.com/user-attachments/assets/5172cdcd-3125-49ee-a8f9-3a4aa8846e2d" width="300">
  <img src="https://github.com/user-attachments/assets/f3703376-6bd9-4bf8-86ef-243d7e6405e5" width="300">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/eb3bf69c-61db-401a-a1b4-70125062a890" width="300">
  <img src="https://github.com/user-attachments/assets/d583b963-0d32-460e-8d24-fa28e64f7383" width="300">
  <img src="https://github.com/user-attachments/assets/60368257-19b0-4c76-90c3-df1409832163" width="300">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/3880ee27-cd78-436e-a0d6-66105a0f3904" width="300">
  <img src="https://github.com/user-attachments/assets/3c8f9ea7-2513-4c5f-8aeb-85cfaecfd592" width="300">
  <img src="https://github.com/user-attachments/assets/49c97fe2-3ad5-498f-902f-a5e1847ebf63" width="300">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/592e12b7-66b7-46fd-9392-38cf94cf67bd" width="300">
  <img src="https://github.com/user-attachments/assets/b54ad01f-70fd-4240-9e5f-90d128350d4d" width="300">
  <img src="https://github.com/user-attachments/assets/117da4bd-6014-44e5-b27c-0712096e8ff1" width="300">
  <img src="https://github.com/user-attachments/assets/0a318733-b89e-456b-a021-99fe4111f26b" width="300">
</p>

### Key Features📌

*   **Intelligent Matching:** Utilizes state-of-the-art NLP models (Transformers, Sentence-Transformers) for semantic similarity scoring.
*   **Scalable Backend:** Implemented as a high-performance **FastAPI** application, ready for deployment and handling concurrent requests.
*   **Modular Architecture:** Follows a clean, organized project structure (`app`, `src`, `models`, `data`, `notebooks`) adhering to best practices for MLOps and software development.
*   **Comprehensive Data Pipeline:** Includes dedicated sections for data processing and model training/evaluation (VS Code Notebooks).
*   **Health and Prediction Endpoints:** Provides a structured API for health checks and single/batch prediction requests.

## 💻 Technology Stack

The project relies on a modern and powerful set of libraries and frameworks:

| Category | Key Technologies | Purpose |
| :--- | :--- | :--- |
| **Backend/API** | `FastAPI`, `Uvicorn` | High-performance asynchronous API service for predictions. |
| **Machine Learning & NLP** | `transformers`, `torch`, `sentence-transformers`, `scikit-learn`, `nltk`, `spacy`, `gensim`, `textblob` | Core libraries for loading and utilizing pre-trained language models , Text preprocessing, feature extraction (TF-IDF), and general NLP tasks. |
| **Data Handling** | `pandas`, `numpy`, `datasets`, `joblib` | Efficient data manipulation, storage, and model persistence. |
| **Development** | `pytest`, `tqdm`, `logging` | Testing, progress tracking, and structured logging. |

## 🏗️ Project Structure

The repository is organized following a standard, scalable structure:

```
AI-Resume-Matcher/
├── app/                      # Main application directory (FastAPI service)
│   ├── api/                  # API routers (e.g., health, predict)
│   ├── services/             # Core business logic (e.g., ModelService for prediction)
│   ├── templates/            # HTML templates for the web interface (if any)
│   ├── utils/                # Application-specific utilities
│   └── main.py               # FastAPI application entry point
├── data/                     # Data storage
│   └── processed/            # Cleaned, processed, and split datasets (train, test, validation)
├── models/                   # Trained model artifacts
│   └── transformer_model     # Directory containing the saved model and tokenizer
├── notebooks/                # Jupyter Notebooks for experimentation, training, and analysis
│   └── *.ipynb               # Training, evaluation, and data exploration notebooks
├── src/                      # Source code for reusable modules (e.g., custom data classes, processors)
├── static/                   # Static files (CSS, JS, images) for the web interface
├── tests/                    # Unit and integration tests
├── requirements.txt          # List of all Python dependencies
└── README.md                 # Project documentation (this file)
```

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites

*   Python 3.9+
*   `pip` package installer

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Menna-Khalid/AI-Resume-Matcher.git
    cd AI-Resume-Matcher
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application (FastAPI)

The application can be run using `uvicorn`, the ASGI server defined in `requirements.txt`.

```bash
uvicorn app.main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`.

*   **Swagger UI (Interactive Docs):** `http://127.0.0.1:8000/api/docs`
*   **Redoc Docs:** `http://127.0.0.1:8000/api/redoc`

### API Endpoints

The core prediction logic is exposed via the `/predictions` endpoint.

| Endpoint | Method | Description | Request Body (Example Fields) |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Checks the service status and model initialization. | None |
| `/predictions/single` | `POST` | Calculates the match score for a single CV/JD pair. | `cv_skill`, `jd_requirement`, `model_type` (optional) |
| `/predictions/batch` | `POST` | Calculates match scores for a batch of CV/JD pairs. | List of single prediction requests. |

## 🧠 Model & Methodology

The matching process is handled by the `ModelService` and is designed to be highly configurable:

1.  **Text Preprocessing:** Resumes and job descriptions are cleaned and processed using standard NLP techniques (`nltk`, `spacy`, `textblob`).
2.  **Feature Extraction:** The text is converted into numerical representations. The system supports:
    *   **TF-IDF Vectorization** (`sklearn.feature_extraction.text.TfidfVectorizer`).
    *   **Sentence Embeddings** (`sentence-transformers`) for deep semantic understanding.
3.  **Similarity Calculation:** The core matching score is determined by calculating the **Cosine Similarity** between the resume vector and the job description vector.
4.  **Model Loading:** The `ModelService` intelligently loads the required model artifacts (e.g., `joblib` for TF-IDF, `transformers` for deep learning models) from the `models/` directory.

The `match_threshold` is set to `0.25` by default, serving as a lower bound for a successful match.

## 📊 Data and Training

The machine learning lifecycle is documented and managed in the `notebooks/` directory.

*   **Data:** The `data/processed` directory contains the datasets used for training, testing, and validation, ensuring reproducibility.
*   **Training:** The Jupyter Notebooks detail the steps for:
    *   Data loading and exploration.
    *   Preprocessing and feature engineering.
    *   Training the chosen NLP models (e.g., fine-tuning a Transformer model for sequence classification or training a Sentence-Transformer).
    *   Model evaluation and selection.

## ✅ Testing

The project includes a dedicated `tests/` directory for ensuring code quality and reliability. Unit tests cover the core logic, model service, and API endpoints.

To run the tests, use `pytest`:

```bash
pytest
```

## 🤝 Contribution

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to open an issue or submit a pull request.

## Thanks☺️!
