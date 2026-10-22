# Text-Based Emotion Detection

A Flask application for text emotion classification with saved logistic regression and SVM models.

## Features

- NLTK text preprocessing
- TF-IDF feature extraction
- Web interface and prediction API

## Technology

Python and Flask.

## Local setup

Clone the repository:

```bash
git clone https://github.com/nasirabbas827/textbasedemotion-ifrah-finalcode.git
cd textbasedemotion-ifrah-finalcode
```

Use a virtual environment:

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
python -m pip install flask joblib scikit-learn nltk
```

The install command lists packages used by the inspected source; this repository does not include a pinned dependency file. Saved machine learning models may require the versions used during training.

Complete the project-specific setup below, then run from the repository root:

```bash
python app.py
```

Open `http://127.0.0.1:5000/`.

## Project-specific setup

Keep the trained model, vectorizer and label encoder files in `models/`. Install the NLTK resources requested by the preprocessing code. Use scikit-learn versions compatible with the saved models.

## Repository layout

| Path | Purpose |
|---|---|
| `NotebookFile/` | Project resources |
| `app.py` | Flask application |
| `models/` | Machine learning code or artifacts |
| `templates/` | HTML templates |

## Development notes

Setup steps describe local development. Configure your own database, mail and API credentials where applicable. Review debug settings and replace development secrets before deploying. The application was not executed as part of this documentation update.

Maintained by [Nasir Abbas](https://github.com/nasirabbas827).
