# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

import json

class tracker_param:
    def __init__(self, source_file=''):
        if source_file:
            with open(source_file) as fid:
                input_param = json.load(fid)
                self.get_param(**input_param)
        else:
            self.get_param()
        
    
    def get_param(self, min_area = 50, max_area = 5000, thresh_C = 15, 
             expected_frames = 15000, fps = 25, compression_buff = 25, roi_size = -1, 
             bad_seg_thresh = 0.5, save_bad_worms = True, 
             thresh_block_size = 61, min_displacement = 0, resampling_N = 49, 
             has_timestamp = True):
        '''
        min_area - minimum area in pixels allowed
        max_area - maximum area in pixels allowed
        thresh_C - threshold used by the adaptative thresholding in the mask calculation
        thresh_block_size - block size used by the adaptative thresholding
        has_timestamp = keep the pixels in the top left corner that correspond to the video timestamp (used only in our setup)
        
        expected_frames - expected number of frames in the video
        fps - frame rate
        roi_size - region of interest size (pixels) used for the skeletonization and individual worm videos. 
        If the value is less than zero, a different size will be estimated for each blob. 
        
        min_displacement - minimum total displacement of a trajectory to be included in the analysis
        bad_seg_thresh - minimum fraction of succesfully skeletonized frames in a worm trajectory to be considered valid
        save_bad_worms - make videos for the bad worm trajectories (under skeletonized)
        
        resampling_N = number of segments used to renormalize the worm skeleton and contours
        '''
        
        if not isinstance(fps, int):
            fps = int(fps)
        
        #backup the input parameters
        self.min_area = min_area
        self.max_area = max_area
        self.thresh_C = thresh_C
        self.expected_frames = expected_frames
        self.fps = fps
        self.bad_seg_thresh = bad_seg_thresh
        self.save_bad_worms = save_bad_worms
        self.thresh_block_size = thresh_block_size
        self.min_displacement = min_displacement
        self.resampling_N = resampling_N
        self.has_timestamp = has_timestamp
        self.compression_buff = compression_buff

        #getROIMask
        self.mask_param = {'min_area': min_area*2, 'max_area': max_area, 'has_timestamp': has_timestamp, 
        'thresh_block_size':thresh_block_size, 'thresh_C':thresh_C}
        
        #compressVideo
        self.compress_vid_param =  {'buffer_size' : compression_buff, 'save_full_interval' : 200*fps, 
                               'max_frame' : 1e32, 
                               'expected_frames':expected_frames, 'mask_param':self.mask_param}
        #getWormTrajectories
        self.get_trajectories_param = {'initial_frame':0, 'last_frame': -1,
                              'min_area':min_area, 'min_length':fps/5, 'max_allowed_dist':fps, 
                              'area_ratio_lim': (0.5, 2), 'buffer_size': compression_buff}
        
        #joinTrajectories
        self.join_traj_param = {'min_track_size': fps*2, 'max_time_gap' : fps*4, 'area_ratio_lim': (0.67, 1.5)}
        
        #getSmoothTrajectories
        self.smoothed_traj_param = {'min_displacement' : min_displacement, 'displacement_smooth_win': fps*4 + 1, 
                               'threshold_smooth_win' : fps*20 + 1, 'roi_size' : roi_size}
        
        #trajectories2Skeletons
        self.get_skeletons_param = {'resampling_N' : resampling_N, 
                               'min_mask_area' : min_area, 'smoothed_traj_param' : self.smoothed_traj_param}
        
        #correctHeadTail
        self.head_tail_param = {'max_gap_allowed' : fps//2, 'window_std' : fps, 'segment4angle' : round(resampling_N/10), 
                           'min_block_size' : fps*10}
        
        #writeIndividualMovies
        self.ind_mov_param = {'fps' : fps, 'bad_seg_thresh' : bad_seg_thresh, 'save_bad_worms': save_bad_worms}
        
        #getWormFeatures
        self.features_param = {'bad_seg_thresh' : bad_seg_thresh, 'fps' : fps}