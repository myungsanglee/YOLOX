python tools/demo_lpd.py image \
-f exps/example/custom/yolox_nano_lpd.py \
-c YOLOX_outputs/yolox_nano_lpd/best_ckpt.pth \
--path /home/fssv2/myungsang/datasets/lpd/green_plate \
--conf 0.25 \
--nms 0.45 \
--device gpu
