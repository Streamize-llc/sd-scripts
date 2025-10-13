
import argparse
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def convert_meta_lora_to_lora(load_path, save_path):
    """
    Converts a Meta-LoRA model (3-layer: down, mid, up) to a standard LoRA model (2-layer: down, up).
    """
    print(f"Loading Meta-LoRA model from: {load_path}")
    meta_lora_sd = load_file(load_path)
    
    converted_sd = {}
    lora_keys = {}

    # Group keys by lora module name
    for key, value in meta_lora_sd.items():
        parts = key.split('.')
        lora_name = parts[0]
        
        if lora_name not in lora_keys:
            lora_keys[lora_name] = {}
            
        if 'lora_down' in key:
            lora_keys[lora_name]['down'] = value
        elif 'lora_mid' in key:
            lora_keys[lora_name]['mid'] = value
        elif 'lora_up' in key:
            lora_keys[lora_name]['up'] = value
        elif 'alpha' in key:
            lora_keys[lora_name]['alpha'] = value
            
    print(f"Found {len(lora_keys)} LoRA modules to convert.")

    for lora_name, keys in tqdm(lora_keys.items(), desc="Converting LoRA modules"):
        if 'down' not in keys or 'mid' not in keys or 'up' not in keys:
            print(f"Skipping {lora_name} due to missing keys.")
            continue

        down_w = keys['down']
        mid_w = keys['mid']
        up_w = keys['up']

        # Perform the conversion: new_up = up @ mid
        # (up_dim, mid_dim) @ (mid_dim, down_dim) -> (up_dim, down_dim)
        # However, LoRA weights are stored as (out_features, in_features)
        # up: (out_features, up_rank)
        # mid: (up_rank, lora_dim)
        # new_up: (out_features, lora_dim)
        new_up_w = up_w @ mid_w

        converted_sd[f"{lora_name}.lora_down.weight"] = down_w
        converted_sd[f"{lora_name}.lora_up.weight"] = new_up_w
        if 'alpha' in keys:
            converted_sd[f"{lora_name}.alpha"] = keys['alpha']

    print(f"Saving converted LoRA model to: {save_path}")
    save_file(converted_sd, save_path)
    print("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the Meta-LoRA model to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the converted standard LoRA model.",
    )
    args = parser.parse_args()

    convert_meta_lora_to_lora(args.load_path, args.save_path)
