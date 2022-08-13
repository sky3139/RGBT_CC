# python3 train.py \
#     --data-dir /home/u20/gitee/RGBTCrowdCounting/RGBT/datasets/cc \
#     --save-dir ./ \
#     --lr 1e-5 \
#     --device 0
python3 train.py \
    --datadir /home/u20/d2/code/RGBTCrowdCounting/DroneRGBT/save/train \
    --save-dir ./datasets \
    --max-epoch 20 \
    --lr 1e-5 \
    --device 0
python3 demo.py \
    --datadir /home/u20/d2/code/RGBTCrowdCounting/DroneRGBT/save/ \
    --save-dir ./datasets \
    --model datasets/0813-182233/best_model.pth \
    --device 0
