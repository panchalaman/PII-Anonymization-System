"""
PII anonymizer using a fine-tuned NER model.

This module provides functionality to anonymize personally identifiable
information in text using a trained named entity recognition model.
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    pipeline
)

class PIIAnonymizer:
    """
    Anonymizes personally identifiable information in text.
    
    This class uses a fine-tuned NER model to detect entities like
    persons, organizations, and locations, and replaces them with
    generic tags or masks.
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the anonymizer with a trained model.
        
        Args:
            model_path: Path to the trained model directory or HuggingFace model ID
            device: Device to run inference on ('cpu', 'cuda'), defaults to best available
        """
        # Set device (use CUDA if available and not explicitly set to CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model from: {model_path}")
        print(f"Device set to use {self.device}")

        try:
            # Try to load from local directory
            if os.path.exists(model_path) and os.path.isdir(model_path):
                # Check for required files
                if not os.path.exists(os.path.join(model_path, "config.json")):
                    print(f"Warning: config.json not found in {model_path}")
                    print("Directory contents:")
                    print(os.listdir(model_path))

                # Load model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)

                # Load label mapping (if available)
                if os.path.exists(os.path.join(model_path, "id2label.json")):
                    with open(os.path.join(model_path, "id2label.json"), "r") as f:
                        self.id2label = json.load(f)
                else:
                    # Fall back to model's config
                    self.id2label = self.model.config.id2label
            else:
                # If local path doesn't exist, try using it as a model ID from HuggingFace Hub
                print(f"Model directory {model_path} not found, attempting to load from HuggingFace Hub")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
                self.id2label = self.model.config.id2label

            # Create NER pipeline for easy inference
            self.nlp = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"  # Merge subword tokens
            )
            print(f"Successfully loaded model and created pipeline")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def anonymize_text(self, text: str, style: str = "tag") -> str:
        """
        Anonymize PII in text by replacing entities with placeholders.
        
        Args:
            text: The text to anonymize
            style: Anonymization style ("tag", "mask", or "redact")
                - tag: Replace with entity type, e.g., [PER]
                - mask: Replace with X characters of same length
                - redact: Replace with [REDACTED]
        
        Returns:
            str: Anonymized text
        """
        if not text:
            return ""
            
        # Detect entities
        entities = self.nlp(text)
        
        # Process in reverse order to avoid index shifting
        anonymized_text = text
        for entity in reversed(entities):
            if entity['entity_group'] != 'O':
                start, end = entity["start"], entity["end"]
                
                # Extract the entity type (remove B-, I- prefixes)
                tag = entity['entity_group'].split('-')[-1] if '-' in entity['entity_group'] else entity['entity_group']
                
                # Apply anonymization based on style
                if style == "tag":
                    replacement = f"[{tag}]"
                elif style == "mask":
                    replacement = "X" * (end - start)
                else:  # redact
                    replacement = "[REDACTED]"
                    
                # Replace entity in text
                anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
                
        return anonymized_text
        
    def detect_entities(self, text: str, threshold: float = 0.7) -> list:
        """
        Detect entities in text with improved boundary detection.
        
        Args:
            text: Text to analyze
            threshold: Confidence threshold for entity detection (0.0 to 1.0)
            
        Returns:
            list: List of detected entities with type, text, and confidence score
        """
        if not text:
            return []

        # Tokenize into words for better boundary detection
        words = text.split()
        
        # Use the tokenizer with word_ids tracking
        inputs = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True
        )
        
        # Get word IDs for mapping subwords back to original words
        word_ids = inputs.word_ids(0)
        
        # Move inputs to device for inference
        device_inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**device_inputs)
        
        # Get predicted labels and probabilities
        predictions = torch.argmax(outputs.logits, axis=2)[0].cpu().numpy()
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=2)[0].cpu().numpy()
        
        # Map predictions back to original words
        word_predictions = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None:  # Skip special tokens
                pred_id = predictions[token_idx]
                prob = probabilities[token_idx, pred_id]
                
                # For each word, keep the highest probability prediction
                if word_idx not in word_predictions or prob > word_predictions[word_idx][1]:
                    word_predictions[word_idx] = (pred_id, prob)
        
        # Extract entities with proper boundaries
        entities = []
        current_entity = None
        
        for i, word in enumerate(words):
            if i in word_predictions:
                pred_id, prob = word_predictions[i]
                # Convert ID to label (handling string/int ID differences)
                pred_id_key = str(pred_id) if isinstance(self.id2label, dict) and str(pred_id) in self.id2label else pred_id
                label = self.id2label[pred_id_key]
                
                if label != "O" and prob >= threshold:
                    # Get entity type without B-/I- prefix
                    entity_type = label.split('-')[-1] if '-' in label else label
                    
                    # Start of a new entity or first entity
                    if label.startswith("B-") or current_entity is None or current_entity["type"] != entity_type:
                        # Save previous entity if exists
                        if current_entity:
                            entities.append(current_entity)
                        
                        # Start new entity
                        current_entity = {
                            "type": entity_type,
                            "text": word,
                            "confidence": float(prob)
                        }
                    # Continue current entity
                    elif label.startswith("I-") and current_entity and current_entity["type"] == entity_type:
                        current_entity["text"] += " " + word
                        # Update confidence as average
                        current_entity["confidence"] = (current_entity["confidence"] + float(prob)) / 2
                    else:
                        # Close current entity
                        if current_entity:
                            entities.append(current_entity)
                        
                        # Start new entity
                        current_entity = {
                            "type": entity_type,
                            "text": word,
                            "confidence": float(prob)
                        }
                elif current_entity:
                    # End of entity
                    entities.append(current_entity)
                    current_entity = None
            elif current_entity:
                # End of entity (word not in predictions)
                entities.append(current_entity)
                current_entity = None
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
