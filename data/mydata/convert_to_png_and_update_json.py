#!/usr/bin/env python3
import os
import json
import argparse
from PIL import Image

def update_transforms_json(json_in, json_out):
    with open(json_in, 'r') as f:
        data = json.load(f)
    for frame in data.get('frames', []):
        base = os.path.splitext(os.path.basename(frame['file_path']))[0]
        frame['file_path'] = f"./images/{base}"
    with open(json_out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated transforms saved to {json_out}")

def convert_jpg_to_png(images_dir):
    for fname in os.listdir(images_dir):
        if fname.lower().endswith('.jpg'):
            base = os.path.splitext(fname)[0]
            jpg = os.path.join(images_dir, fname)
            png = os.path.join(images_dir, f"{base}.png")
            Image.open(jpg).save(png, format='PNG')
            print(f"Converted {fname} -> {base}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_in', required=True, help='Input transforms.json')
    parser.add_argument('--json_out', required=True, help='Output updated JSON')
    parser.add_argument('--images_dir', required=True, help='Directory of images')
    args = parser.parse_args()

    update_transforms_json(args.json_in, args.json_out)
    convert_jpg_to_png(args.images_dir)

if __name__ == "__main__":
    main()
