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
from pathlib import Path

import numpy as np
import json
import sys
import math
import cv2
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place"
    )

    parser.add_argument("--video_in", default="", help="input video to convert to images")
    parser.add_argument("--video_fps", default=2, type=float, help="frame rate for extracting images from video")
    parser.add_argument("--time_slice", default="", help="time slice 't1,t2' for video frames extraction")
    parser.add_argument("--run_colmap", action="store_true", help="run colmap on the image folder")
    parser.add_argument(
        "--colmap_matcher", default="sequential",
        choices=["exhaustive","sequential","spatial","transitive","vocab_tree"],
        help="colmap matcher: sequential for videos, exhaustive for ad-hoc images"
    )
    parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
    parser.add_argument("--images", default="images", help="path to images directory")
    parser.add_argument("--text", default="colmap_text", help="path to colmap text output")
    parser.add_argument(
        "--aabb_scale", default=16, choices=[1,2,4,8,16], type=int,
        help="scene scale factor"
    )
    parser.add_argument("--skip_early", default=0, type=int, help="skip this many images at start")
    parser.add_argument("--out", default="transforms.json", help="output transforms JSON path")
    return parser.parse_args()


def do_system(cmd):
    print(f"==== running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print("FATAL: command failed")
        sys.exit(ret)


def run_ffmpeg(args):
    # prepare images dir
    if not os.path.isabs(args.images):
        args.images = os.path.join(os.path.dirname(args.video_in), args.images)
    images = args.images
    video = args.video_in
    fps = args.video_fps
    print(f"running ffmpeg: video={video}, images dir={images}, fps={fps}")
    if input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip() not in ['y','yes','']:
        sys.exit(1)
    shutil.rmtree(images, ignore_errors=True)
    os.makedirs(images, exist_ok=True)

    # extract frames as PNG
    ts = args.time_slice
    ts_filter = ''
    if ts:
        start, end = ts.split(',')
        ts_filter = f",select='between(t,{start},{end})'"
    cmd = (
        f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps},scale=800:800{ts_filter}\" "
        f"{images}/%04d.png"
    )
    do_system(cmd)


def run_colmap(args):
    db = args.colmap_db
    images = args.images
    base = str(Path(db).with_suffix(''))
    text = args.text if args.text != 'text' else base + '_text'
    sparse = base + '_sparse'
    print(f"running colmap with db={db}, images={images}, sparse={sparse}, text={text}")
    if input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip() not in ['y','yes','']:
        sys.exit(1)
    if os.path.exists(db): os.remove(db)

    # 1) feature extraction (CPU)
    do_system(
        f"colmap feature_extractor --ImageReader.camera_model OPENCV "
        f"--SiftExtraction.estimate_affine_shape=true "
        f"--SiftExtraction.domain_size_pooling=true "
        f"--ImageReader.single_camera=1 "
        f"--SiftExtraction.use_gpu=0 --database_path {db} --image_path {images}"
    )
    # 2) matching (CPU)
    do_system(
        f"colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true "
        f"--SiftMatching.use_gpu=0 --database_path {db}"
    )
    # 3) mapper + BA
    shutil.rmtree(sparse, ignore_errors=True)
    os.makedirs(sparse, exist_ok=True)
    do_system(f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse}")
    do_system(
        f"colmap bundle_adjuster --input_path {sparse}/0 "
        f"--output_path {sparse}/0 --BundleAdjustment.refine_principal_point=1"
    )
    # 4) export TXT
    shutil.rmtree(text, ignore_errors=True)
    os.makedirs(text, exist_ok=True)
    do_system(
        f"colmap model_converter --input_path {sparse}/0 "
        f"--output_path {text} --output_type TXT"
    )


def variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def sharpness(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return variance_of_laplacian(gray)


def qvec2rotmat(q):
    return np.array([
        [1-2*q[2]**2-2*q[3]**2, 2*q[1]*q[2]-2*q[0]*q[3], 2*q[3]*q[1]+2*q[0]*q[2]],
        [2*q[1]*q[2]+2*q[0]*q[3], 1-2*q[1]**2-2*q[3]**2, 2*q[2]*q[3]-2*q[0]*q[1]],
        [2*q[3]*q[1]-2*q[0]*q[2], 2*q[2]*q[3]+2*q[0]*q[1], 1-2*q[1]**2-2*q[2]**2]
    ])

def rotmat(a, b):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    v = np.cross(a, b); c = np.dot(a,b); s = np.linalg.norm(v)
    km = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3) + km + km.dot(km)*((1-c)/(s**2+1e-10))

def closest_point_2_lines(oa, da, ob, db):
    da, db = da/np.linalg.norm(da), db/np.linalg.norm(db)
    c = np.cross(da, db); denom = np.linalg.norm(c)**2
    t = ob-oa; ta = np.linalg.det([t,db,c])/(denom+1e-10)
    tb = np.linalg.det([t,da,c])/(denom+1e-10)
    ta = min(ta, 0); tb = min(tb, 0)
    return ((oa+ta*da+ob+tb*db)*0.5, denom)

if __name__ == "__main__":
    args = parse_args()
    if args.video_in:
        run_ffmpeg(args)
    if args.run_colmap:
        run_colmap(args)

    AABB_SCALE = args.aabb_scale
    SKIP = args.skip_early
    IMG_DIR = args.images
    TXT_DIR = args.text
    OUT = args.out

    print(f"outputting to {OUT}...")
    # read camera params
    with open(os.path.join(TXT_DIR, 'cameras.txt')) as f:
        angle_x = math.pi/2
        for l in f:
            if l.startswith('#'): continue
            e = l.split()
            w, h = float(e[2]), float(e[3])
            fl_x= float(e[4]); fl_y=fl_x
            k1=k2=p1=p2=0; cx, cy = w/2, h/2
            if e[1]=='OPENCV':
                fl_y=float(e[5]); cx=float(e[6]); cy=float(e[7])
                k1, k2, p1, p2 = map(float, e[8:12])
            angle_x=math.atan(w/(fl_x*2))*2
            angle_y=math.atan(h/(fl_y*2))*2
    # assemble JSON
    with open(os.path.join(TXT_DIR, 'images.txt')) as f:
        bottom = np.array([0,0,0,1]).reshape(1,4)
        out = {'camera_angle_x': angle_x, 'camera_angle_y': angle_y,
               'fl_x': fl_x, 'fl_y': fl_y, 'k1': k1, 'k2': k2,
               'p1': p1, 'p2': p2, 'cx': cx, 'cy': cy, 'w': w, 'h': h,
               'aabb_scale': AABB_SCALE, 'frames': []}
        up = np.zeros(3); i=0
        for l in f:
            if l.startswith('#'): continue
            i+=1
            if i<SKIP*2: continue
            if i%2==1:
                elems=l.split()
                fname = '_'.join(elems[9:])
                base, _ = os.path.splitext(fname)
                img_path = os.path.join(IMG_DIR, f"{base}.png")
                b = sharpness(img_path)
                print(img_path, "sharpness=", b)
                q = np.array(list(map(float, elems[1:5])));
                t = np.array(list(map(float, elems[5:8]))).reshape(3,1)
                R = qvec2rotmat(-q); m = np.concatenate([np.concatenate([R, t],1), bottom],0)
                c2w = np.linalg.inv(m)
                c2w[0:3,2]*=-1; c2w[0:3,1]*=-1
                c2w = c2w[[1,0,2,3],:]; c2w[2,:]*=-1
                up+=c2w[0:3,1]
                out['frames'].append({
                    'file_path': f"./images/{base}",
                    'sharpness': b,
                    'transform_matrix': c2w.tolist()
                })
        n = len(out['frames'])
        up/=np.linalg.norm(up)
        R = rotmat(up, [0,0,1]); R = np.pad(R, ([0,1],[0,1])); R[-1,-1]=1
        for fr in out['frames']:
            T = np.array(fr['transform_matrix'])
            fr['transform_matrix'] = (R.dot(T)).tolist()
        # center and scale
        totp=np.zeros(3); totw=0
        for a in out['frames']:
            m = np.array(a['transform_matrix'])[:3,:]
            for bfr in out['frames']:
                mg = np.array(bfr['transform_matrix'])[:3,:]
                p,w = closest_point_2_lines(m[:,3], m[:,2], mg[:,3], mg[:,2])
                if w>0.01: totp+=p*w; totw+=w
        totp/=totw
        for a in out['frames']:
            a['transform_matrix'][0][3]-=totp[0]
            a['transform_matrix'][1][3]-=totp[1]
            a['transform_matrix'][2][3]-=totp[2]
        avg= sum(np.linalg.norm(np.array(a['transform_matrix'])[:3,3]) for a in out['frames'])/n
        for a in out['frames']:
            a['transform_matrix'][0][3]*=4/avg
            a['transform_matrix'][1][3]*=4/avg
            a['transform_matrix'][2][3]*=4/avg
    with open(OUT, 'w') as jf:
        json.dump(out, jf, indent=2)
    print(f"Wrote {n} frames to {OUT}")
