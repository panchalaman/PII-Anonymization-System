"""
Evaluation metrics for NER model performance.

This module calculates common evaluation metrics for named
entity recognition models, such as F1 score, precision, recall, and accuracy.
"""

import numpy as np
from typing import List
from seqeval.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score
)

class PIIMetrics:
    """
    Computes metrics for evaluating NER model performance.
    
    This class uses the seqeval library to compute token-level and
    entity-level metrics like F1-score, which is important for
    evaluating NER models.
    """
    
    def __init__(self, label_list: List[str]):
        """
        Initialize the metrics calculator with the list of possible labels.
        
        Args:
            label_list: List of entity labels (e.g., "O", "B-PER", etc.)
        """
        self.label_list = label_list

    def compute_metrics(self, eval_prediction):
        """
        Compute evaluation metrics from model predictions.
        
        Args:
            eval_prediction: Tuple of (predictions, labels) where:
                - predictions have shape (batch_size, seq_length, num_labels)
                - labels have shape (batch_size, seq_length)
        
        Returns:
            dict: Dictionary of metrics including accuracy, F1, precision, recall
        """
        # Unpack predictions and labels
        predictions, labels = eval_prediction
        
        # Get predicted label IDs (argmax along the class dimension)
        predictions = np.argmax(predictions, axis=2)
        
        # Convert predictions to label names, ignoring special tokens (-100)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Convert reference labels to label names, ignoring special tokens (-100)
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Compute seqeval metrics
        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
        }
