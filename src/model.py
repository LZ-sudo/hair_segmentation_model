"""
SegFormer model for hair segmentation.
"""

import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


class HairSegmentationModel:
    """Wrapper class for SegFormer hair segmentation."""
    
    def __init__(self, model_id="jonathandinu/face-parsing", num_classes=19, device='cuda'):
        """
        Initialize the SegFormer hair segmentation model.
        
        Args:
            model_id: HuggingFace model ID
            num_classes: Number of segmentation classes (for compatibility)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading SegFormer model from {model_id}...")
        
        # Load model and processor from HuggingFace
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        # Get class names/labels from model config
        self.id2label = self.model.config.id2label
        self.num_classes = len(self.id2label)
        
        print(f"SegFormer model loaded successfully!")
        print(f"Number of classes: {self.num_classes}")
        print(f"Available classes: {list(self.id2label.values())}")
        
    def predict(self, image_pil):
        """
        Run inference on PIL Image.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Segmentation logits [1, num_classes, H, W]
        """
        # Ensure we have a PIL Image
        from PIL import Image
        if not isinstance(image_pil, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image_pil)}")
        
        # Store original size for upsampling
        original_size = image_pil.size  # (width, height)
        target_size = (original_size[1], original_size[0])  # (height, width)
        
        with torch.no_grad():
            # Process image using SegFormer's processor
            inputs = self.processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, num_classes, H/4, W/4]
            
            # Upsample to original image size
            logits = torch.nn.functional.interpolate(
                logits,
                size=target_size,  # (height, width)
                mode='bilinear',
                align_corners=False
            )
            
        return logits
    
    def extract_hair_mask(self, logits, hair_class_idx=17):
        """
        Extract hair class from segmentation output.
        
        Args:
            logits: Model output [1, num_classes, H, W]
            hair_class_idx: Index of hair class
            
        Returns:
            Binary mask [H, W] with hair pixels
        """
        # Auto-detect hair class index if the provided one doesn't match
        if hair_class_idx not in self.id2label or 'hair' not in self.id2label[hair_class_idx].lower():
            hair_class_idx = self._find_hair_class_index()
        
        # Get class predictions
        pred = logits.squeeze(0).cpu().numpy()
        pred = pred.argmax(0)  # [H, W]
        
        # Extract hair mask
        hair_mask = (pred == hair_class_idx).astype('uint8')
        return hair_mask
    
    def _find_hair_class_index(self):
        """Find the index of 'hair' class in id2label mapping."""
        for idx, label in self.id2label.items():
            if 'hair' in label.lower():
                print(f"Auto-detected hair class at index {idx}: '{label}'")
                return idx
        
        # Fallback to default
        print("Warning: Could not auto-detect hair class, using index 17")
        return 17