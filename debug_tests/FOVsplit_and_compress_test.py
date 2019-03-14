# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import json
import time
import glob
import numpy as np
import h5py

from sys import platform

from imgstore import new_for_filename
from pprint import pprint


#from tierpsy.analysis.compress.compressVideo import compressVideo
from tierpsy.analysis.compress.processVideo import processVideo
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader


if platform == 'darwin':
    src = "/Users/lferiani/Desktop/Data_FOVsplitter/RawVideos/"
    json_param_file = "/Users/lferiani/Desktop/Data_FOVsplitter/loopbio_rig_new.json"
else:
    raise Exception ("Copy the data locally before starting and modify this script")
    
video_file_list = glob.glob(src + '/*short*/metadata.yaml')


# read parameters
with open(json_param_file) as data_file:    
    json_param = json.load(data_file)
pprint(json_param)

#%%
# name conversion between what's needed by tierpsy's function and what's in the json file, since
# mask_param has more parameters than needed, and they have different names as well
mask_param_f = ['mask_min_area',
                'mask_max_area',
                'thresh_block_size', 
                'thresh_C',
                'dilation_size',
                'keep_border_data',
                'is_light_background']
mask_param = {x.replace('mask_', ''):json_param[x] for x in mask_param_f}

if "mask_bgnd_buff_size" in json_param:
    bgnd_param_mask_f = ['mask_bgnd_buff_size', 
                         'mask_bgnd_frame_gap', 
                         'is_light_background']
    bgnd_param_mask = {x.replace('mask_bgnd_', ''):json_param[x] for x in bgnd_param_mask_f}
else:
    bgnd_param_mask = {}
# if
    
if "save_full_interval" not in json_param: json_param['save_full_interval'] = -1
if "is_extract_timestamp" not in json_param: json_param['is_extract_timestamp'] = False

# put parameters together for processVideo.py
compress_vid_param = {
        'buffer_size': json_param['compression_buff'],
        'save_full_interval': json_param['save_full_interval'],
        'mask_param': mask_param,
        'bgnd_param': bgnd_param_mask,
        'expected_fps': json_param['expected_fps'],
        'microns_per_pixel' : json_param['microns_per_pixel'],
        'is_extract_timestamp': json_param['is_extract_timestamp']
    }                   
                   

# disable background subtraction
compress_vid_param['bgnd_param'] = {}

# set some parameters for background subtraction
#compress_vid_param['bgnd_param'] = {'buff_size': 10, 'frame_gap': 10, 'is_light_background': False}

#%%             

vc = 0;

for video_file in video_file_list[:1]:
    
    # set path variables 
    video_dir, video_name = os.path.split(video_file)
            
    masked_video_dir = video_dir.replace('RawVideos','MaskedVideos')
    masked_image_name = video_name.replace('.yaml','.hdf5')
     
    masked_image_file =  os.path.join(masked_video_dir, masked_image_name)

    
    # clean output folders
    if not os.path.isdir(masked_video_dir):
        os.makedirs(masked_video_dir) 
    if os.path.isfile(masked_image_file):
        os.remove(masked_image_file)
    
            
    tic = time.time()
    
    processVideo(video_file, masked_image_file, compress_vid_param)
    #compressVideo(video_file, masked_image_file, mask_param,  expected_fps=25,
    #                  microns_per_pixel=None, bgnd_param = bgnd_param_mask, buffer_size=-1,
    #                  save_full_interval=-1, max_frame=1e32, is_extract_timestamp=False)
    
    toc = time.time()
    elapsed = toc - tic
    print("Elapsed: %s" % elapsed)
        
    vc += 1
    
#%%
    



