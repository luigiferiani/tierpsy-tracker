
from tierpsy.processing.processMultipleFilesFun import processMultipleFilesFun


from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

with PyCallGraph(output=GraphvizOutput(output_file=)):
    processMultipleFilesFun(analysis_checkpoints=[], 
                        copy_unfinished=False, 
                        end_point='', 
                        force_start_point='', 
                        is_copy_video=False, 
                        is_debug=True, 
                        json_file='/Users/lferiani/Desktop/Data_FOVsplitter/loopbio_rig_new_.json', 
                        mask_dir_root='', 
                        max_num_process=7, 
                        only_summary=True, 
                        pattern_exclude='', 
                        pattern_include='*.yaml', 
                        refresh_time=10.0, 
                        results_dir_root='', 
                        tmp_dir_root='/Users/lferiani/Tmp', 
                        unmet_requirements=False, 
                        video_dir_root='/Users/lferiani/Desktop/Data_FOVsplitter/RawVideos/20190308_48wptest_short_20190308_155935.22436248', 
                        videos_list='')