# PII Anonymization Tool

This project provides tools to detect and anonymize personally identifiable information (PII) in text using Named Entity Recognition (NER). It uses a fine-tuned transformer model based on DistilBERT to identify and replace entities such as person names, organizations, and locations.

## Features

- **NER Model Training**: Fine-tune a transformer model on the CoNLL-2003 dataset
- **Text Anonymization**: Replace detected entities with tags, masks, or redactions
- **Web API**: REST API for text anonymization using Flask
- **Interactive Interface**: User-friendly web interface using Gradio

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pii-anonymization.git
cd pii-anonymization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

The model will be automatically trained when needed. To manually train or retrain:

```python
from main import train_and_test_model

# Train model (will skip if valid model exists)
train_and_test_model("./model")

# Force retraining even if model exists
train_and_test_model("./model", force_train=True)
```

### Using the Anonymizer

```python
from anonymizer import PIIAnonymizer

# Initialize the anonymizer
anonymizer = PIIAnonymizer("./model")

# Anonymize text
text = "John Smith works at Microsoft in New York."
anonymized = anonymizer.anonymize_text(text)
print(anonymized)  # [PER] works at [ORG] in [LOC].

# Detect entities with confidence scores
entities = anonymizer.detect_entities(text)
for entity in entities:
    print(f"{entity['text']} - {entity['type']} ({entity['confidence']:.2%})")
```

### Running the Web API

```bash
python -m app
```

This will start a Flask server on port 5000. You can send POST requests to `/api/anonymize` with a JSON body:

```json
{
  "text": "John Smith works at Microsoft in New York.",
  "threshold": 0.7,
  "style": "tag",
  "include_entities": true
}
```

### Running the Gradio Interface

```bash
python -m gradio_app
```

This will start a web interface that you can access in your browser.

## Project Structure

- `data_processor.py`: Handles loading and processing the CoNLL-2003 dataset
- `tokenizer.py`: Handles tokenization and label alignment for NER
- `metrics.py`: Evaluation metrics for NER model performance
- `trainer.py`: Model training, evaluation, and saving
- `anonymizer.py`: Text anonymization using the trained model
- `app.py`: Flask web API
- `gradio_app.py`: Gradio web interface
- `utils.py`: Helper functions
- `main.py`: Main execution script

## Google Colab

You can also run this project in Google Colab. See the provided notebook for a step-by-step walkthrough.

## Anonymization Styles

The system supports three anonymization styles:

1. **Tag**: Replace entities with their type in brackets, e.g., `[PER]`
2. **Mask**: Replace characters with 'X', maintaining the entity length
3. **Redact**: Replace entities with `[REDACTED]`, regardless of entity type


## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [CoNLL-2003 dataset](https://www.clips.uantwerpen.be/conll2003/ner/)
- https://huggingface.co/datasets/eriktks/conll2003