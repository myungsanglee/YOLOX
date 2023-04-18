python tools/train.py \
-f exps/example/custom/yolox_nano_lpr.py \
-d 0 \
-b 128 \
--fp16 \
-o \
-c yolox_nano.pth \
--cache \
--logger wandb wandb-project lpr