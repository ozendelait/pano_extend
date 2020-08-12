#!/bin/bash
# Create cutouts from all RVC datasets

USE_DATA_ROOT="${RVC_DATA_DIR}"
CUTOUT_OUTP_DIR="${RVC_DATA_DIR}/crops"
COMMON_PARAMS_BOTH="--mask_root ./ --feather_border 5 --max_dim 512"
COMMON_PARAMS_TRAIN="${COMMON_PARAMS_BOTH} --output ${CUTOUT_OUTP_DIR}/train"
COMMON_PARAMS_VAL="${COMMON_PARAMS_BOTH} --output ${CUTOUT_OUTP_DIR}/val"
pushd ${USE_DATA_ROOT}
python3 gen_cutouts.py ${COMMON_PARAMS_TRAIN} --input_json ./wd2_pano.rvc_train.json  --input_root ./wilddash/images/
python3 gen_cutouts.py ${COMMON_PARAMS_VAL} --input_json ./wd2_pano.rvc_val.json --input_root ./wilddash/images/
python3 gen_cutouts.py ${COMMON_PARAMS_TRAIN} --input_json ./cs_pano.rvc_train.json  --input_root ./cityscapes/leftImg8bit/train/
python3 gen_cutouts.py ${COMMON_PARAMS_VAL} --input_json ./cs_pano.rvc_val.json --input_root ./cityscapes/leftImg8bit/val/
python3 gen_cutouts.py ${COMMON_PARAMS_TRAIN} --input_json ./cs_pano.rvc_train.json  --input_root ./cityscapes/leftImg8bit/train/
python3 gen_cutouts.py ${COMMON_PARAMS_VAL} --input_json ./cs_pano.rvc_val.json --input_root ./cityscapes/leftImg8bit/val/
python3 gen_cutouts.py ${COMMON_PARAMS_TRAIN} --input_json ./vider_pano.rvc_train.json  --input_root ./viper/train/img
python3 gen_cutouts.py ${COMMON_PARAMS_VAL} --input_json ./viper_pano.rvc_val.json --input_root ./viper/val/img
python3 gen_cutouts.py ${COMMON_PARAMS_TRAIN} --input_json ./mvd_pano.rvc_train.json --input_root ./mvd/training/images/
python3 gen_cutouts.py ${COMMON_PARAMS_VAL} --input_json ./mvd_pano.rvc_val.json  --input_root ./mvd/validation/images/
python3 gen_cutouts.py ${COMMON_PARAMS_TRAIN} --input_json ./coco_pano.rvc_train.json --input_root ./coco/images/train2017/
python3 gen_cutouts.py ${COMMON_PARAMS_VAL} --input_json ./coco_pano.rvc_val.json --input_root ./coco/images/val2017/
popd