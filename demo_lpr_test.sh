python tools/demo_lpr_test.py image \
-f exps/example/custom/yolox_m_lpr.py \
-c YOLOX_outputs/yolox_m_lpr/best_ckpt.pth \
--path /home/fssv2/myungsang/datasets/lpr/test_v1 \
--conf 0.4 \
--nms 0.5 \
--device gpu
