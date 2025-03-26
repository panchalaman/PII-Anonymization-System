"""
Gradio web interface for PII anonymization.

This module provides an interactive web interface for anonymizing
PII in text using the trained NER model.
"""

import os
import gradio as gr
from anonymizer import PIIAnonymizer

def create_gradio_interface(model_path="./model"):
    """
    Create a Gradio interface for PII anonymization.
    
    Args:
        model_path: Path to the trained model
        
    Returns:
        Gradio Blocks interface
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    anonymizer = PIIAnonymizer(
        model_path=model_path,
        device="cuda" if os.environ.get("USE_CUDA", "").lower() == "true" else "cpu"
    )
    
    def process_text(text, threshold, style, show_entities):
        """
        Process text for the Gradio interface.
        
        Args:
            text: Text to anonymize
            threshold: Confidence threshold
            style: Anonymization style
            show_entities: Whether to show entity details
            
        Returns:
            str: Processed result (anonymized text with optional entity details)
        """
        if not text:
            return "Please enter some text to anonymize."

        try:
            # Detect entities
            entities = anonymizer.detect_entities(text, threshold)
            
            # Anonymize text
            anonymized = anonymizer.anonymize_text(text, style)
            
            # Return result with or without entity details
            if show_entities:
                entity_list = "\n".join([
                    f"• {e['text']} - {e['type']} ({e['confidence']:.2%})"
                    for e in entities
                ])
                return f"Anonymized text:\n{anonymized}\n\nDetected entities:\n{entity_list}"
            else:
                return anonymized

        except Exception as e:
            import traceback
            return f"Error processing text: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

    def highlight_entities(text, threshold=0.7):
        """
        Generate HTML with highlighted entities for visualization.
        
        Args:
            text: Text to analyze
            threshold: Confidence threshold
            
        Returns:
            str: HTML with highlighted entities
        """
        if not text:
            return ""

        # Detect entities
        entities = anonymizer.detect_entities(text, threshold)
        
        # Define entity type colors
        colors = {
            "PER": "#ffcccc",  # Light red for persons
            "ORG": "#ccffcc",  # Light green for organizations
            "LOC": "#ccccff",  # Light blue for locations
            "MISC": "#ffffcc"  # Light yellow for miscellaneous
        }
        
        # Find entity positions in text
        entities_with_pos = []
        for entity in entities:
            start = text.find(entity["text"])
            if start >= 0:
                entities_with_pos.append({
                    "start": start,
                    "end": start + len(entity["text"]),
                    "type": entity["type"],
                    "confidence": entity["confidence"]
                })
        
        # Sort by position reversed (to avoid index shifting)
        entities_with_pos.sort(key=lambda x: x["start"], reverse=True)
        
        # Insert HTML tags for highlighting
        html_text = text
        for entity in entities_with_pos:
            entity_text = text[entity["start"]:entity["end"]]
            entity_type = entity["type"]
            confidence = entity["confidence"]
            color = colors.get(entity_type, "#eeeeee")
            
            # Create highlighted span
            html_entity = (
                f'<span style="background-color: {color};" '
                f'title="{entity_type} ({confidence:.1%})">{entity_text}</span>'
            )
            
            # Replace text with highlighted version
            html_text = html_text[:entity["start"]] + html_entity + html_text[entity["end"]:]
        
        return html_text

    # Create interface with tabs for different functionalities
    with gr.Blocks(title="PII Anonymization Tool") as demo:
        gr.Markdown("# PII Anonymization Tool")
        gr.Markdown("""
        This tool automatically detects and anonymizes personally identifiable information in text
        using a fine-tuned Named Entity Recognition model.
        
        It can detect names, organizations, locations, and other entities in your text.
        """)

        with gr.Tab("Text Anonymization"):
            with gr.Row():
                with gr.Column():
                    # Input area
                    input_text = gr.Textbox(
                        lines=5,
                        placeholder="Enter text to anonymize (e.g., 'John Smith works at Microsoft in New York.')",
                        label="Input Text"
                    )

                    with gr.Row():
                        # Configuration options
                        threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.6,
                            step=0.05,
                            label="Confidence Threshold (higher = fewer replacements)"
                        )
                        style = gr.Radio(
                            ["tag", "mask", "redact"],
                            label="Anonymization Style",
                            value="tag",
                            info="Tag: [PER], Mask: XXXX, Redact: [REDACTED]"
                        )

                    show_entities = gr.Checkbox(
                        label="Show detected entities",
                        value=True
                    )

                    anonymize_btn = gr.Button("Anonymize Text")

                with gr.Column():
                    # Output area
                    output_text = gr.Textbox(label="Result", lines=10)

            # Example inputs
            examples = gr.Examples(
                examples=[
                    ["John Smith works at Microsoft in New York.", 0.6, "tag", True],
                    ["Please contact Sarah Johnson at sarah.j@example.com or call 555-123-4567.", 0.5, "tag", True],
                    ["Patient #12345 was admitted on January 15th with Dr. Williams supervising.", 0.7, "mask", False],
                    ["EU rejects German call to boycott British lamb.", 0.6, "tag", True],
                    ["Apple Inc. announced its new headquarters in Cupertino, California.", 0.7, "tag", True]
                ],
                inputs=[input_text, threshold, style, show_entities]
            )

            # Connect button to processing function
            anonymize_btn.click(
                fn=process_text,
                inputs=[input_text, threshold, style, show_entities],
                outputs=output_text
            )

        with gr.Tab("Entity Highlighting"):
            with gr.Row():
                with gr.Column():
                    # Input area
                    highlight_input = gr.Textbox(
                        lines=5,
                        placeholder="Enter text to highlight entities",
                        label="Input Text"
                    )
                    highlight_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.6,
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    highlight_btn = gr.Button("Highlight Entities")
                with gr.Column():
                    # Output HTML with highlighting
                    highlighted_output = gr.HTML(label="Highlighted Text")

            # Connect button to highlighting function
            highlight_btn.click(
                fn=highlight_entities,
                inputs=[highlight_input, highlight_threshold],
                outputs=highlighted_output
            )

            # Example inputs for highlighting
            highlight_examples = gr.Examples(
                examples=[
                    ["John Smith works at Microsoft in New York.", 0.6],
                    ["EU rejects German call to boycott British lamb.", 0.6],
                    ["The European Commission said on Thursday it disagreed with German advice to consumers to shun British lamb.", 0.7]
                ],
                inputs=[highlight_input, highlight_threshold]
            )

        with gr.Tab("About"):
            # About tab with information about the project
            gr.Markdown("""
            ## About This Tool

            This PII anonymization tool uses a fine-tuned Named Entity Recognition (NER) model 
            based on DistilBERT to identify and anonymize personally identifiable information in text.

            ### Entity Types
            - **PER**: Person names (e.g., "John Smith")
            - **ORG**: Organizations (e.g., "Microsoft", "European Commission")
            - **LOC**: Locations (e.g., "New York", "Brussels")
            - **MISC**: Miscellaneous entities (e.g., "German", "British")

            ### Anonymization Styles
            - **Tag**: Replaces entities with their type in brackets, e.g., [PER]
            - **Mask**: Replaces characters with 'X', maintaining the entity length
            - **Redact**: Replaces entities with [REDACTED], regardless of entity type or length

            ### Model Information
            This tool uses a model fine-tuned on CoNLL-2003 dataset for named entity recognition.
            """)

    return demo

if __name__ == "__main__":
    # Default model path - change as needed
    MODEL_PATH = os.environ.get("MODEL_PATH", "./model")
    
    # Create and launch interface
    demo = create_gradio_interface(MODEL_PATH)
    
    # Launch with public sharing enabled if requested
    share = os.environ.get("SHARE", "").lower() == "true"
    demo.launch(share=share)
