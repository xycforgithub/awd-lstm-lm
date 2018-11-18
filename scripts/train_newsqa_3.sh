CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=x python main.py --data data/newsqa --epochs 750 --dropouti 0.2 --dropouth 0.2 --seed 1023 --save newsqa_3.pt | tee res_newsqa_3_main.txt
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=x python finetune.py --epochs 750 --data data/newsqa --save newsqa_3.pt --dropouti 0.2 --dropouth 0.2 --seed 1023 | tee res_newsqa_3_fine.txt
