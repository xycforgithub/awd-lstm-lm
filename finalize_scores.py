import json
import matplotlib.pyplot as plt
import numpy as np

all_scores = json.load(open('score_marco_a_l.json.json'))
all_ans_scores = [item['answer_score'] for k,item in all_scores.items()]
all_lm_scores = [item['lm_score'] for k,item in all_scores.items()]
all_sum_scores = [item['overall_score'] for k,item in all_scores.items()]
all_ids = [k for k in all_scores]
for i,k in enumerate(all_scores.keys()):
	assert all_ans_scores[i]==all_scores[k]['answer_score']

def normalize_scores(all_sum_scores):
	all_sum_scores = np.array(all_sum_scores)
	all_sum_scores = (all_sum_scores-np.min(all_sum_scores))/(np.max(all_sum_scores)-np.min(all_sum_scores))
	print(np.min(all_sum_scores), np.max(all_sum_scores))
	print(np.mean(all_sum_scores))
	temp_score = np.minimum(all_sum_scores/0.7,1)
	print('0.7',np.mean(temp_score))
	temp_score = np.minimum(all_sum_scores/0.8,1)
	print('0.8',np.mean(temp_score))
	temp_score = np.minimum(all_sum_scores/0.3,1)
	print('0.3',np.mean(temp_score))
	temp_score = np.minimum(all_sum_scores/0.2,1)
	print('0.2',np.mean(temp_score))

	return all_sum_scores

all_sum_scores = normalize_scores(all_sum_scores)
sum_scores = {k:float(v) for k,v in zip(all_ids, all_sum_scores)}
json.dump({'marco':sum_scores},open('sum_scores.json','w'))

all_ans_scores = normalize_scores(all_ans_scores)
ans_scores = {k:float(v) for k,v in zip(all_ids, all_ans_scores)}
json.dump({'marco':ans_scores},open('ans_scores.json','w'))

all_lm_scores = normalize_scores(all_lm_scores)
lm_scores = {k:float(v) for k,v in zip(all_ids, all_lm_scores)}
json.dump({'marco':lm_scores},open('lm_scores.json','w'))