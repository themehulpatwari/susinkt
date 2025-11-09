"""
Fast local image processing using lightweight vision models.
Uses BLIP-base for quick image captioning and context extraction.
"""

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from io import BytesIO
from typing import Union, Optional
import warnings

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class FastImageProcessor:
    """
    Lightweight image processor using BLIP-base model.
    Optimized for speed with model caching and efficient inference.
    """
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the image processor.
        
        Args:
            model_name: HuggingFace model identifier. 
                       'base' variant is fastest, 'large' is more accurate but slower.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model (suppress output)
        self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode for faster inference
    
    def get_image_caption(
        self, 
        image_bytes: bytes, 
        prompt: Optional[str] = None,
        max_length: int = 50
    ) -> str:
        """
        Generate a caption/description for an image.
        
        Args:
            image_bytes: Image data as bytes
            prompt: Optional text prompt to guide captioning (e.g., "a photo of")
            max_length: Maximum length of generated caption
            
        Returns:
            Generated caption/description of the image
        """
        # Load and process image from bytes
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Prepare inputs
        if prompt:
            inputs = self.processor(image, prompt, return_tensors="pt")
        else:
            inputs = self.processor(image, return_tensors="pt")
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3,  # Lower beam search for speed
                early_stopping=True
            )
        
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
    
    def get_image_context(self, image_bytes: bytes) -> dict:
        """
        Get comprehensive context about an image including multiple descriptions.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing various contextual information about the image
        """
        # Generate caption
        caption = self.get_image_caption(image_bytes)
        
        return {
            "description": caption,
            "model": "BLIP-base",
            "device": self.device
        }


def main():
    """Example usage of the FastImageProcessor"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Read image as bytes
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Initialize processor
    processor = FastImageProcessor()
    
    # Get image context
    print("\n" + "="*50)
    print("Processing image...")
    print("="*50)
    
    context = processor.get_image_context(image_bytes)
    
    print(f"\nDescription: {context['description']}")
    print(f"Model: {context['model']}")
    print(f"Device: {context['device']}")


if __name__ == "__main__":
    main()
