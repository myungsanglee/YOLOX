python tools/train.py \
-f exps/example/custom/yolox_darknet_lpr.py \
-d 0 \
-b 64 \
--fp16 \
-o \
-c yolox_darknet.pth \
--cache \
--logger wandb wandb-project lpr \
wandb-name yolox_darknet_lpr