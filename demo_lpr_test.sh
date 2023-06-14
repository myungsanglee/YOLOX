python tools/demo_lpr_test.py image \
-f exps/example/custom/yolox_s_lpr_v4.py \
-c YOLOX_outputs/yolox_s_lpr_v4/epoch_30_ckpt.pth \
--path /home/fssv2/myungsang/datasets/lpd/green_plate \
--conf 0.4 \
--nms 0.5 \
--device gpu
