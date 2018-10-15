CUDA_VISIBLE_DEVICES=1 python main.py --data data/marco --epochs 750 --dropouth 0.2 --seed 1023 --save marco_2.pt
CUDA_VISIBLE_DEVICES=1 python finetune.py --epochs 750 --data data/marco --save marco_2.pt --dropouth 0.2 --seed 1023
