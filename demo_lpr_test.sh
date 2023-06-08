python tools/demo_lpr_test.py image \
-f exps/example/custom/yolox_s_lpr_v5.py \
-c YOLOX_outputs/yolox_s_lpr_v5/best_ckpt.pth \
--path /home/fssv2/myungsang/datasets/lpr/test_v1 \
--conf 0.4 \
--nms 0.5 \
--device gpu
