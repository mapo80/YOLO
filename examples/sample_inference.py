"""
Sample inference script using the YOLO model.

Usage:
    python examples/sample_inference.py --image path/to/image.jpg
    python examples/sample_inference.py --image path/to/image.jpg --weights weights/v9-c.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf

from yolo.config.config import ModelConfig, NMSConfig
from yolo.data.transforms import LetterBox
from yolo.model.yolo import create_model
from yolo.tools.drawer import draw_bboxes
from yolo.utils.bounding_box_utils import Vec2Box, bbox_nms


def load_image(image_path: str, image_size: tuple = (640, 640)):
    """Load and preprocess image."""
    image = Image.open(image_path).convert("RGB")
    letterbox = LetterBox(target_size=image_size)
    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,))}
    image_tensor, _ = letterbox(image, target)
    return image_tensor.unsqueeze(0), image


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")
    parser.add_argument("--model", type=str, default="v9-c", help="Model architecture")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default="output.jpg", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model configuration
    model_yaml = project_root / "yolo" / "config" / "model" / f"{args.model}.yaml"
    model_cfg = OmegaConf.load(model_yaml)
    model_cfg = OmegaConf.merge(OmegaConf.structured(ModelConfig), model_cfg)

    # Create and load model
    weight_path = args.weights if args.weights else True
    model = create_model(model_cfg, weight_path=weight_path, class_num=80)
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    image_tensor, original_image = load_image(args.image)
    image_tensor = image_tensor.to(device)

    # Create converter
    vec2box = Vec2Box(model, model_cfg.anchor, [640, 640], device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_cls, pred_anc, pred_box = vec2box(outputs["Main"])

    # NMS
    nms_cfg = NMSConfig(min_confidence=args.conf, min_iou=args.iou, max_bbox=300)
    predictions = bbox_nms(pred_cls, pred_box, nms_cfg)

    # Draw results
    if len(predictions[0]) > 0:
        pred = predictions[0]
        print(f"Found {len(pred)} objects")

        # Convert to original image coordinates
        result_image = draw_bboxes(original_image, pred, idx2label=None)
        result_image.save(args.output)
        print(f"Result saved to {args.output}")
    else:
        print("No objects detected")


if __name__ == "__main__":
    main()
