CUDA_VISIBLE_DEVICES=1 python eval_probs.py --batch_size 20 --data data/squad --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save squad_1_test.pt
