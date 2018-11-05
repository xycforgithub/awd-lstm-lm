CUDA_VISIBLE_DEVICES=2 python main.py --data data/squad --epochs 750 --dropouti 0.2 --dropouth 0.2 --seed 1023 --save squad_3.pt
CUDA_VISIBLE_DEVICES=2 python finetune.py --epochs 750 --data data/squad --save squad_3.pt --dropouti 0.2 --dropouth 0.2 --seed 1023
