"""
Main execution script for PII anonymization project.

This script provides functions to train the model and run
the anonymization demo. It checks if a model already exists
before training to avoid wasting time.
"""

import os
import shutil
import torch
from data_processor import PIIDataProcessor
from trainer import PIITrainer
from anonymizer import PIIAnonymizer
from utils import (
    save_model_metadata, 
    check_model_exists, 
    get_test_sentences,
    try_fallback_model
)

def train_and_test_model(output_dir="./model", force_train=False):
    """
    Train (if needed) and test a PII anonymization model.
    
    Args:
        output_dir: Directory to save the model to
        force_train: Whether to force training even if model exists
    """
    # --- 1. Check if model already exists ---
    exists, is_valid = check_model_exists(output_dir)
    
    if exists and is_valid and not force_train:
        print(f"✓ Valid trained model already exists at {output_dir}")
        print("  Skipping training to save time.")
        print("  (Use force_train=True to retrain anyway)")
    else:
        if exists and not is_valid:
            print(f"! Found model directory at {output_dir} but it's incomplete or invalid")
            print("  Removing directory and retraining...")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        elif force_train:
            print(f"! Forcing retraining as requested")
            shutil.rmtree(output_dir, ignore_errors=True)
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(f"! No existing model found at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # --- 2. Data Loading and Preparation ---
        print("\n=== Data Loading and Preparation ===")
        processor = PIIDataProcessor()
        datasets = processor.load_conll2003_dataset()
        datasets = processor.convert_to_ner_format(datasets)

        # --- 3. Training ---
        print("\n=== Model Training ===")
        trainer = PIITrainer(
            model_name="distilbert-base-uncased",
            datasets=datasets,
            label_list=processor.label_list,
            id2label=processor.id2label,
            label2id=processor.label2id,
            output_dir=output_dir
        )

        trainer.train()
        
        # --- 4. Evaluation ---
        print("\n=== Model Evaluation ===")
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")
        
        # --- 5. Save Model ---
        print("\n=== Saving Model ===")
        trainer.save_model()
        save_model_metadata(output_dir, processor)
        print(f"Model saved to {output_dir}")

    # --- 6. Test Anonymization ---
    print("\n=== Testing Anonymization ===")
    test_sentences = get_test_sentences()
    
    try:
        print("Initializing anonymizer...")
        anonymizer = PIIAnonymizer(
            model_path=output_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        print("\n===== ANONYMIZATION RESULTS =====\n")
        for sentence in test_sentences:
            anonymized_sentence = anonymizer.anonymize_text(sentence)
            print(f"Original:   {sentence}")
            print(f"Anonymized: {anonymized_sentence}")
            print("-" * 50)
    except Exception as e:
        print(f"\nError in anonymization: {str(e)}")
        try_fallback_model(test_sentences)

def run_web_api(model_dir="./model", port=5000):
    """
    Run the Flask web API.
    
    Args:
        model_dir: Directory containing the trained model
        port: Port to run the API on
    """
    from app import create_flask_app
    
    # Check if model exists
    exists, is_valid = check_model_exists(model_dir)
    if not (exists and is_valid):
        print(f"Error: No valid model found at {model_dir}")
        print("Please train a model first using train_and_test_model()")
        return
    
    # Create and run app
    app = create_flask_app(model_dir)
    app.run(host='0.0.0.0', port=port, debug=True)

def run_gradio_interface(model_dir="./model", share=False):
    """
    Run the Gradio web interface.
    
    Args:
        model_dir: Directory containing the trained model
        share: Whether to create a public share link
    """
    from gradio_app import create_gradio_interface
    
    # Check if model exists
    exists, is_valid = check_model_exists(model_dir)
    if not (exists and is_valid):
        print(f"Error: No valid model found at {model_dir}")
        print("Please train a model first using train_and_test_model()")
        return
    
    # Create and launch interface
    demo = create_gradio_interface(model_dir)
    demo.launch(share=share)

if __name__ == "__main__":
    """
    Main execution script. Uncomment the function you want to run.
    
    - train_and_test_model: Trains a model if needed and tests it
    - run_web_api: Runs the Flask web API
    - run_gradio_interface: Runs the Gradio web interface
    """
    # Set model directory
    MODEL_DIR = "./pii_model"
    
    # Train and test the model (if not already trained)
    train_and_test_model(MODEL_DIR, force_train=False)
    
    # Uncomment to run the Flask web API
    # run_web_api(MODEL_DIR, port=5000)
    
    # Uncomment to run the Gradio web interface
    # run_gradio_interface(MODEL_DIR, share=True)
