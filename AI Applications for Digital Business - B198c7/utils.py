"""
Utility functions for PII anonymization system.

This module provides helper functions for model management,
testing, and demo examples.
"""

import os
import json
import shutil
import torch
from typing import List
from anonymizer import PIIAnonymizer

def save_model_metadata(output_dir, processor):
    """
    Save additional metadata files needed for the model.
    
    Args:
        output_dir: Directory to save files to
        processor: Data processor with label information
    """
    print(f"Saving extra model files to {output_dir}")
    
    # Save label list
    with open(os.path.join(output_dir, "label_list.txt"), "w") as f:
        f.write("\n".join(processor.label_list))
    
    # Save id2label mapping
    with open(os.path.join(output_dir, "id2label.json"), "w") as f:
        json.dump(processor.id2label, f)
    
    # Save label2id mapping
    with open(os.path.join(output_dir, "label2id.json"), "w") as f:
        json.dump(processor.label2id, f)


def check_model_exists(model_dir):
    """
    Check if a trained model exists and is valid.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        tuple: (exists, is_valid) - whether model directory exists and contains required files
    """
    # Check if directory exists
    if not os.path.exists(model_dir):
        return False, False
    
    # Check if directory is not empty
    if not os.listdir(model_dir):
        return True, False
    
    # Check for essential files
    required_files = ["config.json", "pytorch_model.bin"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return True, False
    
    return True, True


def try_fallback_model(test_sentences):
    """
    Try using a fallback model if the custom model fails.
    
    Args:
        test_sentences: List of sentences to test with
    """
    print("\nFallback: Loading a pre-trained NER model from HuggingFace instead")
    try:
        fallback_anonymizer = PIIAnonymizer(
            model_path="dslim/bert-base-NER",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("\n===== FALLBACK ANONYMIZATION RESULTS =====\n")
        for sentence in test_sentences:
            anonymized_sentence = fallback_anonymizer.anonymize_text(sentence)
            print(f"Original:   {sentence}")
            print(f"Anonymized: {anonymized_sentence}")
            print("-" * 50)
    except Exception as fallback_error:
        print(f"\nFallback also failed: {str(fallback_error)}")
        print("Please check your installation of transformers and ensure you have internet connectivity.")


def get_test_sentences():
    """
    Get a list of test sentences for demonstration.
    
    Returns:
        list: Example sentences with different entity types
    """
    return [
        "EU rejects German call to boycott British lamb.",
        "Peter Blackburn works at Microsoft in Seattle.",
        "BRUSSELS 1996-08-22 - The meeting took place at the Grand Hotel.",
        "John Smith and Sarah Johnson attended the conference in New York City last week.",
        "The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb until scientists determine whether mad cow disease can be transmitted to sheep.",
        "Apple Inc. announced its new headquarters in Cupertino, California."
    ]
