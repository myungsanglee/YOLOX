python tools/train.py \
-f exps/example/custom/yolox_s_lpr_v2.py \
-d 1 \
-b 128 \
--fp16 \
-o \
-c yolox_s.pth \
--cache \
--logger wandb wandb-project lpr \
wandb-name yolox_s_lpr_v2