CUDA_VISIBLE_DEVICES=3 python main.py --data data/marco --epochs 750 --dropouti 0.2 --dropouth 0.2 --seed 1023 --save marco_3.pt
CUDA_VISIBLE_DEVICES=3 python finetune.py --epochs 750 --data data/marco --save marco_3.pt --dropouti 0.2 --dropouth 0.2 --seed 1023
