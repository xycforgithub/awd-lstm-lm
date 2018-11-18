import pdb
import json
from math import log
from collections import Counter


test_mode = False
# lm scores
in_domain_score_f = 'lm_scores_s_m.json'
out_domain_score_f = 'lm_scores_m_m.json'
in_data_path = '../data/multitask/san_addmarco_quefea/squad_train.json'
out_data_path = '../data/multitask/san_addmarco_quefea/marco_train.json'
answer_score_on = True
lm_score_on = True
output_name = 'score_marco'

if answer_score_on:
	output_name+='_a'
if lm_score_on:
	output_name+='_l'
output_name +='.json'

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
				answer = qa.get('answers', [])
				questions.append(question)
				answers.append(answer)
				uids.append(uid)
	if test_mode:
		print('using only first 200 data.')
		questions=questions[:200]
		uids=uids[:200]
	return questions,uids, answers

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

pdb.set_trace()
final_scores = {}

for sample in out_data:
	this_score = 0
	try:
		in_answer_score = -log(freq_in[sample['end']-sample['start']+1])
		out_answer_score = -log(freq_out[sample['end']-sample['start']+1])
	except:
		pdb.set_trace()
	answer_score = in_answer_score - out_answer_score
	sample['uid']=str(sample['uid'])
	in_lm_score = in_score[sample['uid']]
	out_lm_score = out_score[sample['uid']]
	lm_score = in_lm_score - out_answer_score

	if answer_score_on:
		this_score+=answer_score
	if lm_score_on:
		this_score+= lm_score

	final_scores[sample['uid']]={'answer_score':answer_score,
	'lm_score':lm_score,
	'overall_score':this_score
	}

dataset_scores = {'marco':final_scores}

json.dump(final_scores, open(output_name+'.json','w'))

print('finish')



