"""
Tokenizer for processing text and aligning labels for NER tasks.

This module handles tokenizing words and aligning entity labels
for transformer-based models like BERT and DistilBERT.
"""

from typing import Dict

class PIITokenizer:
    """
    Tokenizer that handles word to subword alignment for NER tasks.
    
    Transformer models like BERT and DistilBERT use subword tokenization,
    which splits words into smaller pieces. This creates a challenge for
    token classification tasks like NER, where we need to align the labels
    with the subwords.
    """
    
    def __init__(self, tokenizer, label2id: Dict[str, int]):
        """
        Initialize the PII tokenizer.
        
        Args:
            tokenizer: A Hugging Face tokenizer (e.g., AutoTokenizer)
            label2id: Mapping from label names to IDs
        """
        self.tokenizer = tokenizer
        self.label2id = label2id

    def tokenize_and_align_labels(self, examples):
        """
        Tokenize examples and align labels with subword tokens.
        
        This function handles the conversion from word-level labels
        to subword-level labels, using -100 as a special value for
        ignored positions (subword continuations).
        
        Args:
            examples: Batch of examples with tokens and ner_tags
            
        Returns:
            tokenized_inputs: Tokenized examples with aligned labels
        """
        # Tokenize the input words and get word IDs to align labels
        tokenized_inputs = self.tokenizer(
            examples["tokens"],  # List of tokens for each example
            truncation=True,  # Truncate to max length if needed
            is_split_into_words=True,  # Input is already split into words
            padding='max_length',  # Pad to max length
            max_length=128,  # Maximum sequence length
            return_tensors="pt"  # Return PyTorch tensors
        )
        
        # Align labels with subword tokens
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            # Get word IDs for current example
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            
            # Track previous word to handle subword tokens
            previous_word_idx = None
            label_ids = []
            
            # For each token in the sequence:
            for word_idx in word_ids:
                # Special tokens have word_idx = None
                if word_idx is None:
                    label_ids.append(-100)  # Ignore special tokens for loss
                
                # If this is a new word (not a subword continuation)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])  # Use actual label
                
                # If this is a subword continuation
                else:
                    label_ids.append(-100)  # Ignore subword continuations
                
                # Update the previous word index
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        # Add aligned labels to tokenized inputs
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
