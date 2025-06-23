#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")
    parser.add_argument("--video_in", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
    parser.add_argument("--video_fps", default=2)
    parser.add_argument("--time_slice", default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")
    parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
    parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
    parser.add_argument("--images", default="images", help="input path to the images")
    parser.add_argument("--text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
    parser.add_argument("--aabb_scale", default=16, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
    parser.add_argument("--out", default="transforms.json", help="output path")
    args = parser.parse_args()
    return args

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def run_ffmpeg(args):
    if not os.path.isabs(args.images):
        args.images = os.path.join(os.path.dirname(args.video_in), args.images)
    images = args.images
    video = args.video_in
    fps = float(args.video_fps) or 1.0
    print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
    if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
        sys.exit(1)
    try:
        shutil.rmtree(images)
    except:
        pass
    do_system(f"mkdir {images}")

    time_slice_value = ""
    time_slice = args.time_slice
    if time_slice:
        start, end = time_slice.split(",")
        time_slice_value = f",select='between(t\,{start}\,{end})'"
    do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")

def run_colmap(args):
    db = args.colmap_db
    images = args.images
    db_noext = str(Path(db).with_suffix(""))

    if args.text == "text":
        args.text = db_noext + "_text"
    text = args.text
    sparse = db_noext + "_sparse"
    print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
    
    if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip() + "y")[:1] != "y":
        sys.exit(1)
    
    if os.path.exists(db):
        os.remove(db)
    
    # 使用Xvfb来运行COLMAP，以避免图形界面错误
    # Feature extraction
    do_system(f"xvfb-run colmap feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
    
    # Matching
    do_system(f"xvfb-run colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}")
    
    try:
        shutil.rmtree(sparse)
    except:
        pass
    do_system(f"mkdir {sparse}")
    
    # Mapping
    do_system(f"xvfb-run colmap mapper --database_path {db} --image_path {images} --output_path {sparse}")
    
    # Bundle adjustment
    do_system(f"xvfb-run colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
    
    try:
        shutil.rmtree(text)
    except:
        pass
    do_system(f"mkdir {text}")
    
    # Convert model to text format
    do_system(f"xvfb-run colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

# Remaining functions (variance_of_laplacian, sharpness, etc.) stay the same...

if __name__ == "__main__":
    args = parse_args()
    if args.video_in != "":
        run_ffmpeg(args)
    if args.run_colmap:
        run_colmap(args)
    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)
    IMAGE_FOLDER = args.images
    TEXT_FOLDER = args.text
    OUT_PATH = args.out
    print(f"outputting to {OUT_PATH}...")
    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        # Same code for reading and processing camera and images
        pass
    # Final part for writing the JSON file
    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)
