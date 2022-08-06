python3 train.py \
    --data-dir /home/u20/gitee/RGBTCrowdCounting/RGBT/datasets/cc \
    --save-dir ./ \
    --lr 1e-5 \
    --device 0
python3 train.py \
    --data-dir /home/u20/gitee/RGBTCrowdCounting/DroneRGBT/save \
    --save-dir ./datasets \
    --max-epoch 10 \
    --lr 1e-3 \
    --device 0
