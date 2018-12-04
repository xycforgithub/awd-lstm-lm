import json
import pdb
# import matplotlib.pyplot as plt
import numpy as np
import tqdm

dataset_names = ['newsqa','marco']
suffix = '_normb2' # '', '_norm','_inonly','_normsp', '_normb2'
all_score_forms = ['sum','lm','ans']
output_tag = 'nm'

all_scores={}
for dataset_name in dataset_names:
	all_scores[dataset_name]= json.load(open('score_{}_a_l{}.json'.format(dataset_name,suffix)))
test_mode=False

def load_gold_data(path, is_train=True):
	questions = []
	answers=[]
	uids = []
	with open(path, encoding="utf8") as f:
		data = json.load(f)['data']
	cnt = 0
	for article in tqdm.tqdm(data, total=len(data)):
		for paragraph in article['paragraphs']:
			cnt += 1
			for qa in paragraph['qas']:
				uid, question = qa['id'], qa['question']
				answer = [a['text'] for a in qa.get('answers', [])]
				questions.append(question)
				answers.append(answer)
				uids.append(uid)
	if test_mode:
		print('using only first 200 data.')
		questions=questions[:200]
		uids=uids[:200]
	q_dict = {str(uid):{'question':q, 'answer':a} for uid,q,a in zip(uids,questions,answers)}
	return q_dict

for form in all_score_forms:
	print(form,'score')
	if form =='sum':
		input_tag = 'overall_score'
	elif form == 'ans':
		input_tag = 'answer_score'
	else:
		input_tag = 'lm_score'
	min_score = 100
	max_score = -100
	for dataset_name in dataset_names:
		for v in all_scores[dataset_name].values():
			assert np.abs(v['overall_score']-v['answer_score']-v['lm_score'])<0.001
			min_score=min(min_score,v[input_tag])
			max_score = max(max_score, v[input_tag])
	print('min_score=',min_score,'max_score=',max_score)

	out_scores = {}
	all_out_scores_list=[]
	for dataset_name in dataset_names:
		this_scores = {}
		this_scores_list = []
		for k,v in all_scores[dataset_name].items():
			# if form=='sum':
			if v[input_tag]>100:
				pdb.set_trace()
			new_v = 1-(v[input_tag]-min_score)/(max_score-min_score)
			# else:
			# 	new_v = v[input_tag]
			this_scores[k]=new_v
			all_out_scores_list.append(new_v)
			this_scores_list.append(new_v)
		out_scores[dataset_name]=this_scores
		print('dataset {} mean score:'.format(dataset_name), np.mean(this_scores_list))
	output_name='{}{}_{}_scores.json'.format(output_tag,suffix,form)
	json.dump(out_scores,open(output_name,'w'))
	test_values = [1.0,0.8,0.7,0.6,0.5]
	all_out_scores_list = np.array(all_out_scores_list)
	for val in test_values:
		temp_score=np.minimum(all_out_scores_list/val,1)
		print(val,'avg:',np.mean(temp_score))
	
	# if form=='lm':
	q_dict={}
	for dataset_name in dataset_names:
		out_gold_path='../data/{}/train.json'.format(dataset_name)
		q_dict[dataset_name] = load_gold_data(out_gold_path)
	special_count=0
	nonsp_count=0
	special_sum=0.0
	nonsp_sum=0.0

	# for dataset_name in q_dict:
	for dataset_name in ['marco']:
		for k in q_dict[dataset_name]:
			try:
				# if q_dict[dataset_name][k]['question'].split(' ')[0].lower() in ['what','where','who','why','how','which','when','whose','whom']:
				if q_dict[dataset_name][k]['question'].split(' ')[0].lower() in ['when','where']:
					special_count+=1
					special_sum+=out_scores[dataset_name][k]
				else:
					nonsp_count+=1
					nonsp_sum+=out_scores[dataset_name][k]
			except:
				if k not in out_scores[dataset_name]:
					pass
				else:
					pdb.set_trace()
		# print('question:',q_dict[k]['question'],'answers:',q_dict[k]['answer'],
			# 'lm_score:',lm_scores[k],'answer_score:',ans_scores[k],'sum_scores:',sum_scores[k])
		# input()
	print('special_count:',special_count,'special avg:',special_sum/special_count)
	print('nonsp_count:',nonsp_count,'nonsp avg:',nonsp_sum/nonsp_count)
	# pdb.set_trace()

	# if form=='sum':
	# 	norm_scores_lm = json.load(open(output_name.replace('sum','lm')))
	# 	norm_scores_ans = json.load(open(output_name.replace('sum','ans')))

	# 	dataset_name='marco'
	# 	print('mean ans:',np.mean([v for v in norm_scores_ans[dataset_name].values()]))
	# 	print('mean lm:',np.mean([v for v in norm_scores_lm[dataset_name].values()]))
	# 	input()
	# 	for k in q_dict[dataset_name]:

	# 		try:
	# 			print('Q:',q_dict[dataset_name][k]['question'],'A:',q_dict[dataset_name][k]['answer'])
	# 			print('sum score:',out_scores[dataset_name][k],'lm:',norm_scores_lm[dataset_name][k],
	# 				'ans:',norm_scores_ans[dataset_name][k])
	# 			if out_scores[dataset_name][k]<=0.5 or out_scores[dataset_name][k]>=0.8:
	# 				input()
	# 		except:
	# 			if k not in out_scores[dataset_name]:
	# 				pass
	# 			else:
	# 				pdb.set_trace()		



# 	all_ans_scores = [item['answer_score'] for k,item in all_scores.items()]
# 	all_lm_scores = [item['lm_score'] for k,item in all_scores.items()]
# 	all_sum_scores = [item['overall_score'] for k,item in all_scores.items()]
# 	all_ids = [k for k in all_scores]
# 	test_mode=False
# 	for i,k in enumerate(all_scores.keys()):
# 		assert all_ans_scores[i]==all_scores[k]['answer_score']

# def normalize_scores(all_sum_scores):
# 	all_sum_scores = np.array(all_sum_scores)
# 	all_sum_scores = 1-(all_sum_scores-np.min(all_sum_scores))/(np.max(all_sum_scores)-np.min(all_sum_scores))
# 	print(np.min(all_sum_scores), np.max(all_sum_scores))
# 	print(np.mean(all_sum_scores))
# 	temp_score = np.minimum(all_sum_scores/0.8,1)
# 	print('0.8',np.mean(temp_score))
# 	temp_score = np.minimum(all_sum_scores/0.7,1)
# 	print('0.7',np.mean(temp_score))
# 	temp_score = np.minimum(all_sum_scores/0.6,1)
# 	print('0.6',np.mean(temp_score))
# 	temp_score = np.minimum(all_sum_scores/0.5,1)
# 	print('0.5',np.mean(temp_score))

# 	return all_sum_scores

# all_sum_scores = normalize_scores(all_sum_scores)
# sum_scores = {k:float(v) for k,v in zip(all_ids, all_sum_scores)}
# json.dump({'marco':sum_scores},open('sum_scores.json','w'))

# all_ans_scores = normalize_scores(all_ans_scores)
# ans_scores = {k:float(v) for k,v in zip(all_ids, all_ans_scores)}
# json.dump({'marco':ans_scores},open('ans_scores.json','w'))

# all_lm_scores = normalize_scores(all_lm_scores)
# lm_scores = {k:float(v) for k,v in zip(all_ids, all_lm_scores)}
# json.dump({'marco':lm_scores},open('lm_scores.json','w'))

# def load_gold_data(path, is_train=True):
# 	questions = []
# 	answers=[]
# 	uids = []
# 	with open(path, encoding="utf8") as f:
# 		data = json.load(f)['data']
# 	cnt = 0
# 	for article in tqdm.tqdm(data, total=len(data)):
# 		for paragraph in article['paragraphs']:
# 			cnt += 1
# 			for qa in paragraph['qas']:
# 				uid, question = qa['id'], qa['question']
# 				answer = [a['text'] for a in qa.get('answers', [])]
# 				questions.append(question)
# 				answers.append(answer)
# 				uids.append(uid)
# 	if test_mode:
# 		print('using only first 200 data.')
# 		questions=questions[:200]
# 		uids=uids[:200]
# 	q_dict = {str(uid):{'question':q, 'answer':a} for uid,q,a in zip(uids,questions,answers)}
# 	return q_dict

# out_gold_path='../data/marco/train.json'
# q_dict = load_gold_data(out_gold_path)
# special_count=0
# nonsp_count=0
# special_sum=0.0
# nonsp_sum=0.0


# for k in q_dict:
# 	if q_dict[k]['question'].split(' ')[0].lower() in ['what','where','who','why','how','which','when','whose','whom']:
# 		special_count+=1
# 		special_sum+=lm_scores[k]
# 	else:
# 		nonsp_count+=1
# 		nonsp_sum+=lm_scores[k]
# 	# print('question:',q_dict[k]['question'],'answers:',q_dict[k]['answer'],
# 		# 'lm_score:',lm_scores[k],'answer_score:',ans_scores[k],'sum_scores:',sum_scores[k])
# 	# input()
# print('special_count:',special_count,'special avg:',special_sum/special_count)
# print('nonsp_count:',nonsp_count,'nonsp avg:',nonsp_sum/nonsp_count)