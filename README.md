# pano_extend
Quick and dirty experiment to extend label space of panoptic segmentation output by cascading it with a classifier.
Assumes you have a panoptic segmentation framework running which outputs COCO panoptic masks+jsons and a seperate classifier training/prediction pipeline.
The training tool extracts bounding boxes and saves the cutouts into <trg_folder>/<class>/<pano_mask_img_name>_<pano_json_idx>.<ext>
 
