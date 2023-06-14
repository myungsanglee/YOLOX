python tools/train.py \
-f exps/example/custom/yolox_nano_lpd_v2.py \
-d 0 \
-b 32 \
--fp16 \
-o \
-c yolox_nano.pth \
--cache \
--logger wandb wandb-project lpd \
wandb-name yolox_nano_lpd_v2