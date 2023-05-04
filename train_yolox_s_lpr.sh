python tools/train.py \
-f exps/example/custom/yolox_s_lpr.py \
-d 1 \
-b 128 \
--fp16 \
-o \
-c yolox_s.pth \
--cache \
--logger wandb wandb-project lpr