python tools/demo_lpr_test.py image \
-f exps/example/custom/yolox_nano_lpr_rect.py \
-c YOLOX_outputs/yolox_nano_lpr_rect/best_ckpt.pth \
--path /home/fssv2/myungsang/datasets/lpr/test_v1 \
--conf 0.4 \
--nms 0.5 \
--device gpu
