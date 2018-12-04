import pdb
import json
from math import log
from collections import Counter
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--out_dataset_name', type=str, default='newsqa',
                    help='out domain dataset name. in domain is always squad.')
parser.add_argument('--normalize_lm_score', action='store_true', help='normalize lm score.')
parser.add_argument('--out_score_on', action='store_true', help='include scores from out domain.')
parser.add_argument('--norm_score_sp', action='store_true', help='normalize lm score, but only for out domain.')
parser.add_argument('--norm_score_better', action='store_true', 
	help='normalize lm score; norm for each out domain, and norm for in domain over all data.')

args = parser.parse_args()

test_mode = False
out_dataset_name = args.out_dataset_name
# lm scores
if out_dataset_name=='newsqa':
	in_domain_score_f = 'lm_scores_s_n.json'
	out_domain_score_f = 'lm_scores_n_n.json'
else:
	in_domain_score_f = 'lm_scores_s_m.json'
	out_domain_score_f = 'lm_scores_m_m.json'
in_data_path = '../data/multitask/san_addmarco_quefea/squad_train.json'
out_data_path = '../data/multitask/san_addmarco_quefea/{}_train.json'.format(out_dataset_name)
out_gold_path = '../data/{}/train.json'.format(out_dataset_name)
answer_score_on = True
lm_score_on = True
normalize_lm_score = args.normalize_lm_score
norm_score_sp = args.norm_score_sp
norm_score_better = args.norm_score_better
out_score_on = args.out_score_on
output_name = 'score_{}'.format(out_dataset_name)

if answer_score_on:
	output_name+='_a'
if lm_score_on:
	output_name+='_l'
if normalize_lm_score:
	output_name +='_norm'
if norm_score_sp:
	output_name +='_normsp'
if norm_score_better:
	output_name +='_normb2'
if not out_score_on:
	output_name +='_inonly'


# output_name +='.json'

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

def load(path, is_train=True):
	with open(path, 'r', encoding='utf-8') as reader:
		# filter
		data = []
		cnt = 0
		answer_counter=0
		sum_answer_tokens=0
		for line in reader:
			sample = json.loads(line)
			cnt += 1
			if is_train:
				if sample['start'] is None or sample['end'] is None:
					continue
				answer_counter+=1
			data.append(sample)
			if test_mode and len(data)>=200:
				print('use only 200 data.')
				break
	# if is_train and self.span_mode:
		# print('average answer:',sum_answer_tokens/answer_counter,'counter=',answer_counter)
	print('Loaded {} samples out of {}'.format(len(data), cnt))
	return data

in_score = json.load(open(in_domain_score_f))
out_score = json.load(open(out_domain_score_f))

if normalize_lm_score or norm_score_sp or norm_score_better:
	def norm_score(scores,min_score=None, max_score = None):
		if min_score is None:
			min_score = min([v for v in scores.values()])
		if max_score is None:
			max_score = max([v for v in scores.values()])
		for k,v in scores.items():
			scores[k]=(v-min_score)/(max_score-min_score)
		return scores
	if normalize_lm_score:
		in_score=norm_score(in_score)
	elif norm_score_better:
		in_score = norm_score(in_score, min_score=1.0041244983673097, max_score=10.724374532699585)
	# print('lm in socre max:',max([v for v in in_score.values()]),'min:',min([v for v in in_score.values()]))
	# pdb.set_trace()
	out_score = norm_score(out_score)
		


in_data = load(in_data_path)
out_data = load(out_data_path)

all_lengths_s = [sample['end']-sample['start']+1 for sample in in_data]
all_lengths_t = [sample['end']-sample['start']+1 for sample in out_data]

# find a way to count frequency
counter_in = Counter(all_lengths_s)
freq_in = {k:v/len(all_lengths_s) for k,v in counter_in.items()}
counter_out = Counter(all_lengths_t)
freq_out = {k:v/len(all_lengths_t) for k,v in counter_out.items()}
for k in freq_out:
	if k not in freq_in:
		freq_in[k]=1/len(all_lengths_s)

# pdb.set_trace()
final_scores = {}
q_dict = load_gold_data(out_gold_path)

in_answer_score={}
out_answer_score={}
for sample in out_data:
	sample['uid']=str(sample['uid'])
	try:
		in_answer_score[sample['uid']] = -log(freq_in[sample['end']-sample['start']+1])
		out_answer_score[sample['uid']] = -log(freq_out[sample['end']-sample['start']+1])
	except:
		pdb.set_trace()
if norm_score_better:
	in_answer_score = norm_score(in_answer_score,max_score=11.38052486133421, min_score=1.163358596320627)
	out_answer_score = norm_score(out_answer_score)
	print('answer in socre max:',max([v for v in in_answer_score.values()]),'min:',min([v for v in in_answer_score.values()]))

for sample in out_data:
	this_score = 0
	in_lm_score = in_score[sample['uid']]
	out_lm_score = out_score[sample['uid']]
	if not out_score_on:
		lm_score = in_lm_score
		answer_score = in_answer_score[sample['uid']]
	else:
		lm_score = in_lm_score - out_lm_score
		answer_score = in_answer_score[sample['uid']] - out_answer_score[sample['uid']]

	if answer_score_on:
		this_score+=answer_score
	if lm_score_on:
		this_score+= lm_score

	final_scores[sample['uid']]={'answer_score':answer_score,
	'lm_score':lm_score,
	'overall_score':this_score
	}
	k=sample['uid']
	# print('question:',q_dict[k]['question'],'answers:',q_dict[k]['answer'],
	# 	'lm_score:',final_scores[k]['lm_score'],'answer_score:',final_scores[k]['answer_score'],
	# 	'in_lm_score:',in_lm_score,'out_lm_score:',out_lm_score)
	# pdb.set_trace()

dataset_scores = {'marco':final_scores}

json.dump(final_scores, open(output_name+'.json','w'))

print('finish')
pdb.set_trace()
for k in final_scores:
	print('question:',q_dict[k]['question'],'answers:',q_dict[k]['answer'],
		'lm_score:',final_scores[k]['lm_score'],'answer_score:',final_scores[k]['answer_score'])
	input()

