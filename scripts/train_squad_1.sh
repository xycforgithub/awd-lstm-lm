# CUDA_VISIBLE_DEVICES=1 python main.py --batch_size 20 --data data/squad --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save squad_1.pt
CUDA_VISIBLE_DEVICES=1 python finetune.py --batch_size 20 --data data/squad --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save squad_1.pt
# CUDA_VISIBLE_DEVICES=1 python pointer.py --data data/squad --save squad_1_test.pt --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000