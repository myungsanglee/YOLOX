python tools/train.py \
-f exps/example/custom/yolox_s_lpr_v5.py \
-d 0 \
-b 128 \
--fp16 \
-o \
--cache \
--logger wandb wandb-project lpr \
wandb-name yolox_s_lpr_v5
