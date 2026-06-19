"""
Baseline Defense: OpenAI Privacy Filter
This script uses the open-weight OpenAI Privacy Filter 
to redact sensitive PII tokens from documents *before* summarization.
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class PrivacyFilterDefense:
    def __init__(self, model_id: str = "openai/privacy-filter", device: str = "cuda"):
        print(f"Loading Privacy Filter model: {model_id} on {device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Load the MoE token classification model
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True
            )
            
            # Create a pipeline for NER (Named Entity Recognition) / token classification
            self.nlp = pipeline(
                "token-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple" # Aggregates adjacent tokens of the same class
            )
        except Exception as e:
            print(f"Error initializing Privacy Filter: {e}")
            self.nlp = None
        
    def redact(self, text: str) -> str:
        """
        Detects PII in the text and replaces it with a generic <REDACTED> tag.
        """
        if not self.nlp:
            print("Privacy filter not loaded. Returning original text.")
            return text
            
        # The filter will identify PII spans like names, emails, phone numbers.
        entities = self.nlp(text)
        
        # Sort entities by start index in reverse order to avoid offset shifting when replacing
        redacted_text = text
        for entity in sorted(entities, key=lambda x: x['start'], reverse=True):
            start = entity['start']
            end = entity['end']
            entity_type = entity.get('entity_group', 'PII')
            
            # Replace the PII with a placeholder tag
            placeholder = f"<{entity_type.upper()}>"
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]
            
        return redacted_text

if __name__ == "__main__":
    # Quick demonstration
    sample_text = "Bệnh nhân Nguyễn Văn A, sinh năm 1980, số điện thoại 0912345678, địa chỉ tại 123 Đường B, Quận 1, TP.HCM."
    
    print("Initializing OpenAI Privacy Filter Baseline...")
    defense = PrivacyFilterDefense(device="cpu") # Use CPU for quick local testing
    
    if defense.nlp:
        safe_text = defense.redact(sample_text)
        print("\n--- Original Document ---")
        print(sample_text)
        print("\n--- Scrubbed Document ---")
        print(safe_text)
