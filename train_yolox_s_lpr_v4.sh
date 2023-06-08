python tools/train.py \
-f exps/example/custom/yolox_s_lpr_v4.py \
-d 0 \
-b 128 \
--fp16 \
-o \
--cache \
--resume \
--logger wandb wandb-project lpr \
wandb-name yolox_s_lpr_v4