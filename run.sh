# CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 20 --data data/squad --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save marco_temp.pt
# CUDA_VISIBLE_DEVICES=0 python eval_probs.py --batch_size 20 --data data/marco --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save marco_1.pt --batch_size 10 --original_data data/marco --save_name m_m
# CUDA_VISIBLE_DEVICES=0 python eval_probs.py --batch_size 20 --data data/marco --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save squad_1.pt --batch_size 10 --original_data data/squad --save_name s_m
# CUDA_VISIBLE_DEVICES=0 python eval_probs.py --batch_size 20 --data data/newsqa --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save squad_1.pt --batch_size 10 --original_data data/squad --save_name s_n
# CUDA_VISIBLE_DEVICES=0 python eval_probs.py --batch_size 20 --data data/newsqa --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save newsqa_1.pt --batch_size 10 --original_data data/newsqa --save_name n_n

# python generate_scores.py --out_dataset_name marco --out_score_on --norm_score_better
python generate_scores.py --out_dataset_name newsqa --out_score_on --norm_score_better
# python generate_scores.py --out_dataset_name newsqa --out_score_on --norm_score_sp

# CUDA_VISIBLE_DEVICES=2 python finetune.py --batch_size 20 --data data/squad --dropouti 0.4 --dropouth 0.25 --seed 1023 --epoch 600 --save squad_1.pt