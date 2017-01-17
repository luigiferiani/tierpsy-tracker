# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:44:07 2015

@author: ajaver
"""

import json
import os
from MWTracker import AUX_FILES_DIR

#deprecated variables that will be ignored
deprecated_fields = ['has_timestamp', 'min_displacement']

#the dict key are the old names and the value the new name
deprecated_alias = {
    'fps': 'expected_fps',
    'threshold_factor': 'worm_bw_thresh_factor',
    'is_invert_thresh' : 'is_light_background',
    'is_fluorescence' : 'is_light_background',
    'min_length' : 'min_box_width',
    'max_area' : 'mask_max_area'}

dflt_param_list = [
    ('mask_min_area', 50, 'minimum allowed area in pixels allowed for in the compression mask.'),
    ('mask_max_area', int(1e8), 'maximum allowed area in pixels allowed for in the compression mask..'),
    ('min_box_width', 5, 'minimum allowed width of bounding box in pxels.'),
    ('thresh_C', 15, 'constant offset used by the adaptative thresholding to calculate the mask.'),
    ('thresh_block_size', 61, 'block size used by the adaptative thresholding.'),
    ('dilation_size', 9, 'size of the structural element used in morphological operations to calculate the worm mask.'),
    ('compression_buff', 25, 'number of images "min-averaged" to calculate the image mask.'),
    ('keep_border_data', False, 'set it to false if you want to remove any connected component that touches the border.'),
    ('is_light_background', True, 'set to true to indentify dark worms over a light background.'),
    ('expected_fps', 25, 'expected frame rate.'),
    ('traj_min_area', 25, 'minimum allowed area in pixels allowed for the trajectories and the videos.'),
    ('traj_max_allowed_dist', 25, 'Maximum displacement expected between frames to be consider same track.'),
    ('traj_area_ratio_lim', [0.5, 2], 'Limits of the consecutive blob areas to be consider the same object.'),
    ('worm_bw_thresh_factor', 1.05, 'This factor multiplies the threshold used to binarize the individual worms image.'),
    ('resampling_N', 49, 'number of segments used to renormalize the worm skeleton and contours.'),
    ('max_gap_allowed_block', 10, 'maximum time gap allowed between valid skeletons to be considered as belonging in the same group. Head/Tail correction by movement.'),
    ('strel_size', 5, 'Structural element size. Used to calculate skeletons and trajectories.'),
    ('fps_filter', -1, 'frame per second used to calculate filters for trajectories. As default it will have the same value as expected_fps. Set to zero to eliminate filtering.'),
    ('filt_bad_seg_thresh', 0.8, 'minimum fraction of succesfully skeletonized frames in a worm trajectory to be considered valid'),
    ('filt_min_displacement', 10, 'minimum total displacement of a trajectory to be used to calculate the threshold to dectect bad skeletons.'),
    ('filt_critical_alpha', 0.01, 'critical chi2 alpha used in the mahalanobis distance to considered a worm a global outlier.'),
    ('filt_max_width_ratio', 2.25, 'Maximum width radio between midbody and head or tail. Does the worm more than double its width from the head/tail? Useful for coils.'),
    ('filt_max_area_ratio', 6, 'maximum area ratio between head+tail and the rest of the body to be a valid worm.  Are the head and tail too small (or the body too large)?'),
    ('save_int_maps', False, 'save the intensity maps and not only the profile along the worm major axis.'),
    ('int_avg_width_frac', 0.3, 'width fraction from the center used to calculate the worm axis intensity profile.'),
    ('int_width_resampling', 15, 'width in pixels of the intensity maps'),
    ('int_length_resampling', 131, 'length in pixels of the intensity maps'),
    ('int_max_gap_allowed_block', -1, 'maximum time gap allowed between valid intensity maps to be considered as belonging in the same group. Head/Tail correction by intensity.'),
    ('split_traj_time', 300, 'time in SECONDS that a trajectory will be subdivided to calculate the splitted features.'),
    ('roi_size', -1, ''),
    ('filter_model_name', '', ''),
    ('n_cores_used', 1, 'Number of core used. Currently it is only suported by TRAJ_CREATE and it is only recommended at high particle densities.'),
    #not tested (used for the zebra fish)
    ('use_background_subtraction', False, 'Flag to determine whether background should be subtracted from original frames.'),
    ('background_threshold', 1, 'Threshold value to use when applying background subtraction.'),
    ('ignore_mask', False, 'Mask is not used if set to True. Only applies if background subtraction is also active.'),
    ('background_type', 'GENERATE_DYNAMICALLY', 'If background subtraction is enabled, determines whether background is generated dynamically or loaded from a file'),
    ('background_frame_offset', 500, 'Number of frames offset from current frame used for generating background images.'),
    ('background_generation_function', 'MAXIMUM', 'Function to apply to frames in order to generate background images.'),
    ('background_file', '', 'Image file to use for background subtraction. If a filepath is set here, this file will be used instead of dynamically-generated background images.'),
    ('analysis_type', 'WORM', 'Analysis functions to use.'),
    ('zf_num_segments', 12, 'Number of segments to use in tail model.'),
    ('zf_min_angle', -90, 'The lowest angle to test for each segment. Angles are set relative to the angle of the previous segment.'),
    ('zf_max_angle', 90, 'The highest angle to test for each segment.'),
    ('zf_num_angles', 90, 'The total number of angles to test for each segment. Eg., If the min angle is -90 and the max is 90, setting this to 90 will test every 2 degrees, as follows: -90, -88, -86, ...88, 90.'),
    ('zf_tail_length', 60, 'The total length of the tail model in pixels.'),
    ('zf_tail_detection', 'MODEL_END_POINT', 'Algorithm to use to detect the fish tail point.'),
    ('zf_prune_retention', 1, 'Number of models to retain after scoring for each round of segment addition. Higher numbers will be much slower.'),
    ('zf_test_width', 2, 'Width of the tail in pixels. This is used only for scoring the model against the test frame.'),
    ('zf_draw_width', 2, 'Width of the tail in pixels. This is used for drawing the final model.'),
    ('zf_auto_detect_tail_length', True, 'Flag to determine whether zebrafish tail length detection is used. If set to True, values for zf_tail_length, zf_num_segments and zf_test_width are ignored.')
]

#separate parameters default data into dictionaries for values and help
default_param = {x: y for x, y, z in dflt_param_list}
param_help = {x: z for x, y, z in dflt_param_list}


class tracker_param:

    def __init__(self, source_file=''):
        input_param = default_param.copy()
        if source_file:
            with open(source_file) as fid:
                param_in_file = json.load(fid)
            for key in param_in_file:

                if key in deprecated_fields:
                    #ignore deprecated fields
                    continue
                elif key in deprecated_alias:
                    #rename deprecated alias
                    input_param[deprecated_alias[key]] = param_in_file[key]
                
                elif key == 'min_area':
                    #special case of deprecated alias
                    input_param['mask_min_area'] = param_in_file['min_area']
                    input_param['traj_min_area'] = param_in_file['min_area']/2

                else:
                    input_param[key] = param_in_file[key]

        self._get_param(**input_param)
        # print(input_param)

    def _get_param(
            self,
            mask_min_area,
            mask_max_area,
            min_box_width,
            thresh_C,
            thresh_block_size,
            dilation_size,
            compression_buff,
            keep_border_data,
            is_light_background,
            expected_fps,
            traj_max_allowed_dist,
            traj_area_ratio_lim,
            traj_min_area,
            worm_bw_thresh_factor,
            resampling_N,
            max_gap_allowed_block,
            fps_filter,
            strel_size,
            filt_bad_seg_thresh,
            filt_min_displacement,
            filt_critical_alpha,
            filt_max_width_ratio,
            filt_max_area_ratio,
            save_int_maps,
            int_avg_width_frac,
            int_width_resampling,
            int_length_resampling,
            int_max_gap_allowed_block,
            split_traj_time,
            use_background_subtraction,
            ignore_mask,
            background_type,
            background_threshold,
            background_frame_offset,
            background_generation_function,
            background_file,
            analysis_type,
            zf_num_segments,
            zf_min_angle,
            zf_max_angle,
            zf_num_angles,
            zf_tail_length,
            zf_tail_detection,
            zf_prune_retention,
            zf_test_width,
            zf_draw_width,
            zf_auto_detect_tail_length,
            filter_model_name,
            roi_size,
            n_cores_used):

        if not isinstance(expected_fps, int):
            expected_fps = int(expected_fps)

        self.expected_fps = expected_fps

        # getROIMask
        self.mask_param = {
            'min_area': mask_min_area,
            'max_area': mask_max_area,
            'thresh_block_size': thresh_block_size,
            'thresh_C': thresh_C,
            'dilation_size': dilation_size,
            'keep_border_data': keep_border_data,
            'is_light_background': is_light_background}

        # compressVideo
        self.compress_vid_param = {
            'buffer_size': compression_buff,
            'save_full_interval': 200 * expected_fps,
            'max_frame': 1e32,
            'mask_param': self.mask_param,
            'expected_fps': expected_fps,
            'use_background_subtraction': use_background_subtraction,
            'ignore_mask': ignore_mask,
            'background_type': background_type,
            'background_threshold': background_threshold,
            'background_frame_offset': background_frame_offset,
            'background_generation_function': background_generation_function,
            'background_file': background_file}

        # parameters for a subsampled video
        self.subsample_vid_param = {
            'time_factor' : 8, 
            'size_factor' : 5, 
            'expected_fps' : 30
        }

        # getWormTrajectories
        self.trajectories_param = {
            'buffer_size' : compression_buff,
            'min_area': traj_min_area,
            'min_box_width': min_box_width,
            'worm_bw_thresh_factor': worm_bw_thresh_factor,
            'strel_size': (strel_size, strel_size),
            'analysis_type' : analysis_type,
            'thresh_block_size':thresh_block_size,
            'n_cores_used': 1}

        # joinTrajectories
        min_track_size = max(1, fps_filter * 2)
        max_time_gap = max(0, fps_filter * 4)
        self.join_traj_param = {
            'max_allowed_dist': traj_max_allowed_dist,
            'min_track_size': min_track_size,
            'max_time_gap': max_time_gap,
            'area_ratio_lim': traj_area_ratio_lim}

        # getSmoothTrajectories
        self.smoothed_traj_param = {
            'min_track_size': min_track_size,
            'displacement_smooth_win': expected_fps + 1,
            'threshold_smooth_win': expected_fps * 20 + 1,
            'roi_size': roi_size}

        if filter_model_name:
            assert os.path.exists(os.path.join(AUX_FILES_DIR, filter_model_name))
        self.init_skel_param = {
            'smoothed_traj_param': self.smoothed_traj_param,
            'filter_model_name' : filter_model_name
            }

        self.blob_feats_param =  {
            'is_light_background' : is_light_background,
            'strel_size' : strel_size
            }

        zf_skel_args = {'zf_num_segments': zf_num_segments,
            'zf_min_angle': zf_min_angle,
            'zf_max_angle': zf_max_angle,
            'zf_num_angles': zf_num_angles,
            'zf_tail_length': zf_tail_length,
            'zf_tail_detection': zf_tail_detection,
            'zf_prune_retention': zf_prune_retention,
            'zf_test_width': zf_test_width,
            'zf_draw_width': zf_draw_width,
            'zf_auto_detect_tail_length': zf_auto_detect_tail_length}


        # trajectories2Skeletons
        self.skeletons_param = {
            'resampling_N': resampling_N,
            'worm_midbody': (0.33, 0.67),
            'min_blob_area': traj_min_area,
            'strel_size': strel_size,
            'analysis_type': analysis_type,
            'zf_skel_args' : zf_skel_args}
            

        # correctHeadTail
        if max_gap_allowed_block < 0:
            expected_fps // 2
        self.head_tail_param = {
            'max_gap_allowed': max_gap_allowed_block,
            'window_std': expected_fps,
            'segment4angle': round(
                resampling_N / 10),
            'min_block_size': expected_fps * 10}

        min_num_skel = 4 * expected_fps
        # getWormFeatures
        self.feat_filt_param = {
            'min_num_skel': min_num_skel,
            'bad_seg_thresh': filt_bad_seg_thresh,
            'min_displacement': filt_min_displacement,
            'critical_alpha': filt_critical_alpha,
            'max_width_ratio': filt_max_width_ratio,
            'max_area_ratio': filt_max_area_ratio}

        self.int_profile_param = {
            'width_resampling': int_width_resampling,
            'length_resampling': int_length_resampling,
            'min_num_skel': min_num_skel,
            'smooth_win': 11,
            'pol_degree': 3,
            'width_percentage': int_avg_width_frac,
            'save_int_maps': save_int_maps}

        if int_max_gap_allowed_block < 0:
            int_max_gap_allowed_block = max_gap_allowed_block / 2

        smooth_W = max(1, round(expected_fps / 5))
        self.head_tail_int_param = {
            'smooth_W': smooth_W,
            'gap_size': int_max_gap_allowed_block,
            'min_block_size': (
                2 * expected_fps // 5),
            'local_avg_win': 10 * expected_fps,
            'min_frac_in': 0.85,
            'head_tail_param': self.head_tail_param}

        self.feats_param = {
            'expected_fps': expected_fps, 
            'feat_filt_param': self.feat_filt_param,
            'split_traj_time' : split_traj_time
        }

if __name__=='__main__':
    json_file = '/Users/ajaver/Documents/GitHub/Multiworm_Tracking/fluorescence/pharynx.json'
    params = tracker_param(json_file)
    print(params.trajectories_param)

