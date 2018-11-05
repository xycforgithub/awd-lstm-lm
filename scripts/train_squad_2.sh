CUDA_VISIBLE_DEVICES=1 python main.py --data data/squad --epochs 750 --dropouth 0.2 --seed 1023 --save squad_2.pt
CUDA_VISIBLE_DEVICES=1 python finetune.py --epochs 750 --data data/squad --save squad_2.pt --dropouth 0.2 --seed 1023
