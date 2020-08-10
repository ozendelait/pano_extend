#!/usr/bin/env python
# -*- coding: utf-8 -*-
# create classifier cutoputs from panoptic JSON

import os
import sys
import argparse
import json
import tqdm
import numpy as np
import cv2

opencv_version = cv2.__version__ if hasattr(cv2, '__version__') else cv2.__file__.replace('\\','').replace('/','').split("cv2-")[-1].split("-")[0]
if len(opencv_version) < 4 or int(opencv_version[0]) < 3 or float(opencv_version[0:3]) < 3.2:
    print("Warning: Use opencv version > 3.1 to prevent Jpeg rotation problems!") 

def bgr2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 2] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 0]
    return int(color[2] + 256 * color[1] + 256 * 256 * color[0])

#recursively add files, ignore case of ext.
def recursive_add_ic(folder, ext, mapping, max_depth = 7):
    if ext[0] != '.':
        ext = '.'+ext
    for f in os.listdir(folder):
        fullp = os.path.join(folder,f)
        if os.path.isfile(fullp):
            filename_ext = os.path.splitext(f)[1].lower()
            if filename_ext  == ext:
                mapping.append(fullp)
        elif f != '.' and f.find('..') < 0 and max_depth > 0:
            recursive_add_ic(os.path.join(folder,f), ext, mapping, max_depth-1)

def fname_path(path):
    return os.path.splitext(os.path.basename(path))[0].replace('_leftImg8bit','').replace('_gtFine_panoptic','')

def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, 
                        help="Input COCO annotation json")
    parser.add_argument('--mask_root', type=str, default = None,
                        help="Root folder with COCO masks pngs; per default in subfolder <json_name>/*")
    parser.add_argument('--input_root', type=str, 
                        help="Input root folder for intensity images (prob. RGB)")
    parser.add_argument('--output', type=str,
                        help="Output root folder for cutouts")
    parser.add_argument('--ext', type=str, default = "jpg",
                        help="Output file extension for cutouts")
    parser.add_argument('--min_area', type=int, default = 15,
                        help="Min num pixel in mask for valid segments (remainder are skipped)")
    parser.add_argument('--categories', type=str, default = None,
                        help="Optional category name file (COCO or mapillary config.json style")
    parser.add_argument('--feather_border', type=int, default = 2,
                        help="Num of pixel to feather when masking non-instance labels (after resizing!)")
    parser.add_argument('--max_dim', type=int, default = 512,
                        help="Scale output cutouts to this max dim")

    args = parser.parse_args(argv)
    print("Loading COCO annotation file " + args.input_json + "...")
    with open(args.input_json, 'r') as ifile:
        annot = json.load(ifile)
    
    cat_names = {}
    if not args.categories is None:
        with open(args.categories, 'r') as ifile:
            cat_j = json.load(ifile)
    else:
        cat_j = annot #annotation file might include "categories" entry
    #create category_id -> category_name dict (default name is always the category_id itself)
    instance_labels = {}
    if isinstance(cat_j, dict):
        for entry_name, idx_start, inst_name in [("categories",0,"isthing"),("labels",1,"instances")]:
            if not entry_name in cat_j:
                continue
            for idx, cat in enumerate(cat_j[entry_name]):
                cid = int(cat.get('id',idx+idx_start))
                cat_names[cid] = cat["name"].replace(' ','_') #MVD is 1-indexed! -> add 1 to list idx for category id
                if inst_name in cat and cat[inst_name]:
                    instance_labels[cid] = True
            break
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.mask_root is None:
      mask_dir = os.path.join(os.path.dirname(os.path.realpath(args.input_json)),os.path.splitext(os.path.basename(args.input_json))[0])
    else:
      mask_dir = os.path.realpath(args.mask_root)
    
    if not os.path.exists(mask_dir):
        print("Error: no panoptic input masks at "+mask_dir)
        return -2

    inp_img_paths = []
    recursive_add_ic(args.input_root, 'jpg', inp_img_paths, max_depth = 3)
    if len(inp_img_paths) == 0:
        recursive_add_ic(args.input_root, 'jpeg', inp_img_paths, max_depth = 3)
    if len(inp_img_paths) == 0:
        recursive_add_ic(args.input_root, 'png', inp_img_paths, max_depth = 3)
    img_name_mapping = {fname_path(p):p for p in inp_img_paths}
        
    #first sanity check all paths
    for a in annot['annotations']:
        inp_mask_path = os.path.join(mask_dir, a['file_name'])
        if not os.path.exists(inp_mask_path):
            print("Warning: no mask image at "+inp_mask_path)
            continue
        fname = fname_path(a['file_name'])
        inp_img_path = img_name_mapping.get(fname,None)
        if inp_img_path is None or not os.path.exists(inp_img_path):
            print("Warning: no input image at ",inp_img_path," for mask "+fname)
            continue
    feather_border = args.feather_border 
    if args.feather_border > 0:
        feather_border = (args.feather_border)*2+1
    #iterate through annotations
    for a in tqdm.tqdm(annot['annotations']):
        fname = fname_path(a['file_name'])
        inp_mask_path = os.path.join(mask_dir, a['file_name'])
        mask_ids = bgr2id(cv2.imread(inp_mask_path))
        inp_img = cv2.imread(img_name_mapping[fname])
        if inp_img.shape[:2] != mask_ids.shape[:2]:
            print("Warning: input image and mask mismatch: ",img_name_mapping[fname], inp_img.shape[:2]," vs. mask "+fname+ " with ",mask_ids.shape[:2])
            continue
            
        for idx, segment in enumerate(a['segments_info']):
            trg_path = os.path.join(args.output, str(cat_names.get(segment["category_id"], segment["category_id"])), fname+"_%i.%s"%(idx, args.ext))
            if os.path.exists(trg_path):
                continue
            if not os.path.exists(os.path.dirname(trg_path)):
                os.makedirs(os.path.dirname(trg_path))
            segm_mask = mask_ids == segment['id']
            #check area at mask
            area = np.count_nonzero(segm_mask)
            if area < args.min_area:
                continue
            #calculate bounding box for cutout
            segm_minmax = np.where(segm_mask)
            if len(segm_minmax) < 1 or len(segm_minmax[0]) == 0:
                continue
            rmin, rmax, cmin, cmax = np.min(segm_minmax[0]), np.max(segm_minmax[0]), np.min(segm_minmax[1]), np.max(segm_minmax[1])
            cutout = inp_img[rmin:rmax, cmin:cmax,:]
            if feather_border>= 0 and not instance_labels.get(segment["category_id"],False): #need to mask non-class labels
                cutout = np.copy(cutout)
                mask_cutout = mask_ids[rmin:rmax, cmin:cmax]
                #feather borders
                if feather_border > 0:
                    mask_feather = np.zeros_like(cutout)
                    mask_feather[mask_cutout==segment['id']] = 255
                    mask_feather = cv2.GaussianBlur(mask_feather,(feather_border,feather_border),0)
                    cutout = cutout * (mask_feather/255)
                else:
                    cutout[mask_cutout!=segment['id']] = 0
            if args.max_dim > 0 and max(cutout.shape[0], cutout.shape[1]) > args.max_dim:
                scaling = args.max_dim/max(cutout.shape[0], cutout.shape[1])
                cutout=cv2.resize(cutout,None,fx=scaling, fy=scaling, interpolation = cv2.INTER_AREA)

            cv2.imwrite(trg_path,cutout)
    return 0
    
if __name__ == "__main__":
    sys.exit(main())