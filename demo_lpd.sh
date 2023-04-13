python tools/demo_lpd.py image \
-f exps/example/custom/yolox_nano_lpd.py \
-c YOLOX_outputs/yolox_nano_lpd/best_ckpt.pth \
--path /home/fssv2/myungsang/datasets/lpd/coco_format/v1/val2017/data_02 \
--conf 0.25 \
--nms 0.45 \
--device gpu
