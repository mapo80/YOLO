convert_dict = {
    "19.cv1": "19.conv",
    "16.cv1": "16.conv",
    ".7.cv1": ".7.conv",
    ".5.cv1": ".5.conv",
    ".3.cv1": ".3.conv",
    ".28.": ".29.",
    ".25.": ".26.",
    ".22.": ".23.",
    "cv": "conv",
    ".m.": ".bottleneck.",
}

HEAD_NUM = "29"


def convert_weight(old_state_dict, new_state_dict, model_size: int = 38):
    new_weight_set = set(new_state_dict.keys())
    for weight_name, weight_value in old_state_dict.items():
        if HEAD_NUM in weight_name:
            _, _, conv_name, conv_id, *post_fix = weight_name.split(".")
            head_id = 30 if conv_name in ["cv2", "cv3"] else 22
            head_type = "anchor_conv" if conv_name in ["cv2", "cv4"] else "class_conv"
            weight_name = ".".join(["model", str(head_id), "heads", conv_id, head_type, *post_fix])
        else:
            for old_name, new_name in convert_dict.items():
                if old_name in weight_name:
                    weight_name = weight_name.replace(old_name, new_name)
        if weight_name in new_weight_set:
            assert new_state_dict[weight_name].shape == weight_value.shape, f"shape miss match {weight_name}"
            new_state_dict[weight_name] = weight_value
            new_weight_set.remove(weight_name)

    return new_state_dict


head_converter = {
    "head_conv": "m",
    "implicit_a": "ia",
    "implicit_m": "im",
}

SPP_converter = {
    "pre_conv.0": "cv1",
    "pre_conv.1": "cv3",
    "pre_conv.2": "cv4",
    "post_conv.0": "cv5",
    "post_conv.1": "cv6",
    "short_conv": "cv2",
    "merge_conv": "cv7",
}

REP_converter = {"conv1": "rbr_dense", "conv2": "rbr_1x1", "conv": "0", "bn": "1"}


def convert_weight_v7(old_state_dict, new_state_dict):
    map_weight = []
    for key_name in new_state_dict.keys():
        new_shape = new_state_dict[key_name].shape
        old_key_name = "model." + key_name
        new_key_name = key_name
        if old_key_name not in old_state_dict.keys():
            if "heads" in key_name:
                layer_idx, _, conv_idx, conv_name, *details = key_name.split(".")
                old_key_name = ".".join(["model", str(layer_idx), head_converter[conv_name], conv_idx, *details])
            elif (
                "pre_conv" in key_name
                or "post_conv" in key_name
                or "short_conv" in key_name
                or "merge_conv" in key_name
            ):
                for key, value in SPP_converter.items():
                    if key in key_name:
                        key_name = key_name.replace(key, value)
                old_key_name = "model." + key_name
            elif "conv1" in key_name or "conv2" in key_name:
                for key, value in REP_converter.items():
                    if key in key_name:
                        key_name = key_name.replace(key, value)
                old_key_name = "model." + key_name
        map_weight.append(old_key_name)
        assert old_key_name in old_state_dict.keys(), f"Weight Name Mismatch!! {old_key_name}"
        old_shape = old_state_dict[old_key_name].shape
        assert new_shape == old_shape, "Weight Shape Mismatch!! {old_key_name}"
        new_state_dict[new_key_name] = old_state_dict[old_key_name]
    return new_state_dict


replace_dict = {"cv": "conv", ".m.": ".bottleneck."}


def convert_weight_seg(old_state_dict, new_state_dict):
    diff = -1
    for old_weight_name in old_state_dict.keys():
        old_idx = int(old_weight_name.split(".")[1])
        if old_idx == 23:
            diff = 3
        elif old_idx == 41:
            diff = -19
        new_idx = old_idx + diff
        new_weight_name = old_weight_name.replace(f".{old_idx}.", f".{new_idx}.")
        for key, val in replace_dict.items():
            new_weight_name = new_weight_name.replace(key, val)

        if new_weight_name not in new_state_dict.keys():
            heads = "heads"
            _, _, conv_name, conv_idx, *details = old_weight_name.split(".")
            if "proto" in conv_name:
                conv_idx = "3"
                new_weight_name = ".".join(["model", str(layer_idx), heads, conv_task, *details])
                continue
            if "dfl" in old_weight_name:
                continue
            if conv_name == "cv2" or conv_name == "cv3" or conv_name == "cv6":
                layer_idx = 44
                heads = "detect.heads"
            if conv_name == "cv4" or conv_name == "cv5" or conv_name == "cv7":
                layer_idx = 25
                heads = "detect.heads"

            if conv_name == "cv2" or conv_name == "cv4":
                conv_task = "anchor_conv"
            if conv_name == "cv3" or conv_name == "cv5":
                conv_task = "class_conv"
            if conv_name == "cv6" or conv_name == "cv7":
                conv_task = "mask_conv"
                heads = "heads"

            new_weight_name = ".".join(["model", str(layer_idx), heads, conv_idx, conv_task, *details])

        if (
            new_weight_name not in new_state_dict.keys()
            or new_state_dict[new_weight_name].shape != old_state_dict[old_weight_name].shape
        ):
            print(f"new: {new_weight_name}, old: {old_weight_name}")
            print(f"{new_state_dict[new_weight_name].shape} {old_state_dict[old_weight_name].shape}")
        new_state_dict[new_weight_name] = old_state_dict[old_weight_name]
    return new_state_dict


import sys
from pathlib import Path

import hydra
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import BaseModel


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: Config):
    old_weight_path = getattr(cfg, "old_weight", "v9t.pt")
    new_weight_path = getattr(cfg, "new_weight", "ait.pt")
    print(f"Changing {old_weight_path} -> {new_weight_path}")
    cfg.weight = None
    model = BaseModel(cfg)
    old_weight = torch.load(old_weight_path, weights_only=False)
    new_weight = convert_weight(old_weight, model.model.state_dict())
    model.model.load_state_dict(new_weight)
    torch.save(model.model.model.state_dict(), new_weight_path)
    cfg.weight = new_weight_path
    BaseModel(cfg)


if __name__ == "__main__":
    main()
