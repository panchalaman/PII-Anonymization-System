"""
Flask web API for PII anonymization.

This module provides a REST API for anonymizing PII in text
using the trained NER model.
"""

import os
from flask import Flask, request, jsonify, render_template
from anonymizer import PIIAnonymizer

def create_flask_app(model_path="./model"):
    """
    Create and configure Flask app for PII anonymization.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Flask app ready to run
    """
    # Initialize app
    app = Flask(__name__)
    
    # Load the model
    anonymizer = PIIAnonymizer(model_path, "cuda" if os.environ.get("USE_CUDA", "").lower() == "true" else "cpu")
    
    @app.route('/')
    def home():
        """Render the home page with API documentation."""
        return render_template('index.html')

    @app.route('/api/anonymize', methods=['POST'])
    def api_anonymize():
        """
        API endpoint for text anonymization.
        
        POST parameters:
        - text: The text to anonymize
        - threshold: Confidence threshold (0.0-1.0), default 0.7
        - style: Anonymization style ("tag", "mask", "redact"), default "tag"
        - include_entities: Whether to include entity details in response, default False
        
        Returns:
            JSON with original and anonymized text, and optionally entities
        """
        # Get request data
        data = request.json
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        # Get parameters
        text = data['text']
        threshold = float(data.get('threshold', 0.7))
        style = data.get('style', 'tag')
        include_entities = data.get('include_entities', False)

        try:
            # Anonymize text
            anonymized_text = anonymizer.anonymize_text(text, style)
            
            # Prepare response
            response = {
                "original_text": text,
                "anonymized_text": anonymized_text,
            }
            
            # Include entities if requested
            if include_entities:
                entities = anonymizer.detect_entities(text, threshold)
                response["entities"] = entities
                
            return jsonify(response)
            
        except Exception as e:
            import traceback
            return jsonify({
                "error": str(e), 
                "traceback": traceback.format_exc()
            }), 500

    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint to verify API is running."""
        return jsonify({
            "status": "healthy", 
            "model": model_path
        })
    
    # Create template directory and index page
    create_template_files()
    
    return app

def create_template_files():
    """Create necessary template files for the Flask app."""
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html template
    with open('templates/index.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>PII Anonymization API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
                .endpoint { margin-bottom: 20px; }
                code { background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>PII Anonymization API</h1>
            <p>This API provides endpoints for anonymizing personally identifiable information (PII) in text.</p>

            <div class="endpoint">
                <h2>Anonymize Text</h2>
                <p><strong>Endpoint:</strong> <code>POST /api/anonymize</code></p>
                <p><strong>Request body:</strong></p>
                <pre>
{
  "text": "John Smith works at Microsoft in New York.",
  "threshold": 0.7,
  "style": "tag",
  "include_entities": true
}
                </pre>
                <p><strong>Response:</strong></p>
                <pre>
{
  "original_text": "John Smith works at Microsoft in New York.",
  "anonymized_text": "[PER] works at [ORG] in [LOC].",
  "entities": [
    {"type": "PER", "text": "John Smith", "confidence": 0.98},
    {"type": "ORG", "text": "Microsoft", "confidence": 0.95},
    {"type": "LOC", "text": "New York", "confidence": 0.94}
  ]
}
                </pre>
            </div>

            <div class="endpoint">
                <h2>Health Check</h2>
                <p><strong>Endpoint:</strong> <code>GET /api/health</code></p>
                <p><strong>Response:</strong></p>
                <pre>
{
  "status": "healthy",
  "model": "./model"
}
                </pre>
            </div>
        </body>
        </html>
        ''')

if __name__ == '__main__':
    # Default model path - change as needed
    MODEL_PATH = os.environ.get("MODEL_PATH", "./model")
    
    # Create and run app
    app = create_flask_app(MODEL_PATH)
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port, debug=True)
