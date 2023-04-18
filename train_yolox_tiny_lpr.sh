python tools/train.py \
-f exps/example/custom/yolox_tiny_lpr.py \
-d 0 \
-b 128 \
--fp16 \
-o \
-c yolox_tiny.pth \
--cache \
--logger wandb wandb-project lpr \
wandb-name yolox_tiny_lpr