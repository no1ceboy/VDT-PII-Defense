"""
Baseline Defense: OpenAI Privacy Filter
This script uses the open-weight OpenAI Privacy Filter 
to redact sensitive PII tokens from documents *before* summarization.
"""

import torch

class PrivacyFilterDefense:
    def __init__(self, model_id: str = "openai/privacy-filter", device: str = "cuda"):
        print(f"Loading Privacy Filter model: {model_id} on {device}...")
        self.runtime = None
        try:
            from opf._common.checkpoint_download import ensure_default_checkpoint
            from opf._core.runtime import load_inference_runtime
            from opf._core.decoding import ViterbiCRFDecoder
        except ImportError:
            print("\nError: The official OPF library is not installed.")
            print("Please run: pip install git+https://github.com/openai/privacy-filter.git\n")
            return
            
        try:
            print("Ensuring OPF checkpoint is downloaded (this may take a moment on first run)...")
            checkpoint_path = ensure_default_checkpoint()
            
            # The model_id argument is not strictly used by opf as it pulls from its default,
            # but we allow passing it for API consistency.
            self.runtime = load_inference_runtime(
                checkpoint=checkpoint_path,
                device_name=device,
                trim_span_whitespace=True,
                discard_overlapping_predicted_spans=True,
                output_mode="typed"
            )
            # Create a decoder for use during inference
            self.decoder = ViterbiCRFDecoder(self.runtime.label_info)
            print("OPF Runtime loaded successfully!")
        except Exception as e:
            print(f"Error initializing Privacy Filter: {e}")
            self.runtime = None
        
    def redact(self, text: str) -> str:
        """
        Detects PII in the text and replaces it with a generic <REDACTED> tag.
        """
        if not self.runtime:
            print("Privacy filter not loaded. Returning original text.")
            return text
            
        from opf._core.runtime import predict_text
        
        # Predict PII spans
        result = predict_text(self.runtime, text, decoder=self.decoder)
        
        # Sort spans by start index in reverse order to avoid offset shifting when replacing
        redacted_text = text
        for span in sorted(result.spans, key=lambda x: x.start, reverse=True):
            start = span.start
            end = span.end
            entity_type = span.label
            
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
