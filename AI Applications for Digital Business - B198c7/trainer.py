"""
Trainer for the PII anonymization model.

This module handles training, evaluation, and saving of the
named entity recognition model for PII detection.
"""

import torch
from typing import Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

from tokenizer import PIITokenizer
from metrics import PIIMetrics

class PIITrainer:
    """
    Trainer for the NER model used for PII detection.
    
    This class handles setting up, training, evaluating, and saving
    a transformer-based token classification model.
    """
    
    def __init__(self, model_name: str, datasets, label_list: List[str], 
                 id2label: Dict, label2id: Dict, output_dir: str = "./pii-model"):
        """
        Initialize the trainer with model and data.
        
        Args:
            model_name: Hugging Face model identifier (e.g., "distilbert-base-uncased")
            datasets: Dataset with train, validation splits
            label_list: List of entity labels
            id2label: Mapping from label IDs to names
            label2id: Mapping from label names to IDs
            output_dir: Directory to save the model to
        """
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Initialize tokenizer and model
        print(f"Initializing tokenizer and model from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pii_tokenizer = PIITokenizer(self.tokenizer, label2id)
        
        # Tokenize the datasets
        print("Tokenizing datasets...")
        self.tokenized_datasets = datasets.map(
            self.pii_tokenizer.tokenize_and_align_labels, 
            batched=True, 
            remove_columns=datasets["train"].column_names
        )
        
        # Initialize model with the correct number of labels
        print(f"Initializing model with {len(label_list)} labels...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=len(label_list), 
            id2label=id2label, 
            label2id=label2id
        )
        
        # Set up metrics calculator
        self.metrics = PIIMetrics(label_list)
        
        # Configure training arguments
        use_cuda = torch.cuda.is_available()
        device_str = "CUDA" if use_cuda else "CPU"
        print(f"Using {device_str} for training")
        
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            fp16=use_cuda,  # Use FP16 if CUDA is available
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="none",
            logging_dir='./logs',
            logging_steps=100,
        )
        
        # Initialize the trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self.metrics.compute_metrics,
        )

    def train(self):
        """
        Train the model.
        
        Returns:
            TrainOutput: Training statistics and metrics
        """
        print("Starting model training...")
        return self.trainer.train()

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model...")
        return self.trainer.evaluate()

    def save_model(self):
        """
        Save the model, tokenizer, and configuration files.
        
        Saves all necessary files for later loading the model.
        """
        print(f"Saving model to {self.output_dir}...")
        
        # Save model using the trainer (saves optimizer state too)
        self.trainer.save_model(self.output_dir)
        
        # Explicitly save tokenizer and config
        self.tokenizer.save_pretrained(self.output_dir)
        self.model.config.save_pretrained(self.output_dir)
        
        print("Model saved successfully!")
