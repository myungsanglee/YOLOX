python tools/eval.py \
-f exps/example/custom/yolox_s_lpr.py \
-c YOLOX_outputs/yolox_s_lpr/best_ckpt.pth \
-d 1 \
-b 128 \
--conf 0.001 \
# --fp16 \
# --fuse