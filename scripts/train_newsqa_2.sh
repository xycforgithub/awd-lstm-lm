CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=x python main.py --data data/newsqa --epochs 750 --dropouth 0.2 --seed 1023 --save newsqa_2.pt | tee res_newsqa_2_main.txt
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=x python finetune.py --epochs 750 --data data/newsqa --save newsqa_2.pt --dropouth 0.2 --seed 1023 | tee res_newsqa_2_fine.txt
