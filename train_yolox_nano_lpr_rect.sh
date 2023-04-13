python tools/train.py \
-f exps/example/custom/yolox_nano_lpr_rect.py \
-d 1 \
-b 128 \
--fp16 \
-o \
-c yolox_nano.pth \
--cache \
--logger wandb wandb-project lpr