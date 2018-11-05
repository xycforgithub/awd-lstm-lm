import json
from math import log
from collections import Counter
test_mode = True
# lm scores
source_domain_score_f = 'lm_scores_m_m.json'
target_domain_score_f = 'lm_scores_s_m.json'
source_data_path = '../data/multitask/san_addmarco_quefea/marco_train.json'
target_data_path = '../data/multitask/san_addmarco_quefea/squad_train.json'
answer_score_on = True
lm_score_on = True
output_name = 'score_marco'

if answer_score_on:
	output_name+='_a'
if lm_score_on:
	output_name+='_l'
output_name +='.json'

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

source_score = json.load(open(in_domain_score_f))
target_score = json.load(open(out_domain_score_f))

source_data = load(source_data_path)
target_data = load(target_data_path)

all_lengths_s = [sample['end']-sample['start']+1 for sample in source_data]
all_lengths_t = [sample['end']-sample['start']+1 for sample in target_data]

# find a way to count frequency
counter_source = Counter(training_data)
freq_source = {k:v/len(all_lengths_s) for k,v in counter_source.items()}
counter_target = Counter(target_data)
freq_target = {k:v/len(all_lengths_t) for k,v in counter_target.items()}

final_scores = {}

for sample in training_data:
	this_score = 0
	source_answer_score = -log(freq_source[sample['end']-sample['start']+1])
	target_answer_score = -log(freq_target[sample['end']-sample['start']+1])

	source_lm_score = source_score[sample['uid']]
	target_lm_score = target_score[sample['uid']]

	if answer_score_on:
		this_score+=source_lm_score
	if lm_score_on:
		this_score++TODO

	final_scores[sample['uid']]=this_score

dataset_scores = {'marco':final_scores}

json.dump(final_scores, open(output_name,'w'))





