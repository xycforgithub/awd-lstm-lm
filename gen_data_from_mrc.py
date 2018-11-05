import json
import spacy
import argparse
import re
import tqdm
import unicodedata
import pdb
from collections import Counter
import random
import string

test_mode=False
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str, default='../data/', help='dataset folder.')
parser.add_argument('--dataset_name',type=str, default='squad')
parser.add_argument('--output_path', type=str, default='data/')
args = parser.parse_args()

def normalize_text(text):
	# try:
	return unicodedata.normalize('NFD', text)
	# except:
		# pdb.set_trace()
def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '
def reform_text(text):
	text = re.sub(u'-|¢|¥|€|£|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
	text = text.strip(' \n')
	text = re.sub('\s+', ' ', text)
	return text

def load_data(path, is_train=True):
	questions = []
	uids = []
	with open(path, encoding="utf8") as f:
		data = json.load(f)['data']
	cnt = 0
	for article in tqdm.tqdm(data, total=len(data)):
		for paragraph in article['paragraphs']:
			cnt += 1
			for qa in paragraph['qas']:
				uid, question = qa['id'], qa['question']
				questions.append(question)
				uids.append(uid)
	if test_mode:
		print('using only first 200 data.')
		questions=questions[:200]
		uids=uids[:200]
	return questions,uids

if args.dataset_name=='squad':
	process_orders=['train','dev','dev']
else:
	process_orders=['train','dev','test']

output_names = ['train','valid','test']
word_counter=Counter()
nlp = spacy.load('en', disable=['vectors', 'textcat', 'tagger', 'ner', 'parser'])
all_questions = []
all_questions_tokened = []
all_uids = []

for part_name,out_name in zip(process_orders,output_names):
	questions,uids = load_data(args.data_path+args.dataset_name+'/{}.json'.format(part_name))
	# remove duplicates
	id_map = {}
	for uid,question in zip(uids,questions):
		id_map[uid]=question
	uids=[]
	questions=[]
	for uid,question in id_map.items():
		uids.append(uid)
		questions.append(question)
	print('after removing duplicates, len=',len(questions))

	questions = [reform_text(question) for question in questions]
	questions_tokened = [q_doc for q_doc in nlp.pipe(questions, batch_size=10000, n_threads=32)]
	all_questions_tokened.append(questions_tokened)
	all_questions.append(questions)
	all_uids.append(uids)
	for tokened in tqdm.tqdm(questions_tokened, total=len(questions)):
		# pdb.set_trace()
		word_counter.update([normalize_text(w.lower_) for w in tokened if len(normalize_text(w.lower_)) > 0])
vocab = sorted([w for w in word_counter], key=word_counter.get, reverse=True)
print('size of vocab:',len(vocab))
print('appear more than 10 times:', len([w for w in word_counter if word_counter.get(w)>=10]))
# pdb.set_trace()
try:
	vocab=vocab[:10000]
except:
	pdb.set_trace()


for part_name,out_name, questions, uids, questions_tokened in zip(process_orders,
	output_names, all_questions, all_uids, all_questions_tokened):
	random_order = [i for i in range(len(questions))]
	random.shuffle(random_order)
	uids = [uids[idx] for idx in random_order]
	questions = [questions[idx] for idx in random_order]
	questions_tokened = [questions_tokened[idx] for idx in random_order]
	# pdb.set_trace()


	add_str='_test' if test_mode else ''
	fout_txt=open(args.output_path+args.dataset_name+add_str+'/{}.txt'.format(out_name),'w', encoding='utf-8')
	fout_uid=open(args.output_path+args.dataset_name+add_str+'/{}_id.txt'.format(out_name),'w', encoding='utf-8')
	nonpunc_count=0
	for tokened,uid in tqdm.tqdm(zip(questions_tokened, uids), total=len(questions)):
		question_text = [normalize_text(w.lower_) for w in tokened if len(normalize_text(w.lower_)) > 0]
		question_text = [w if w in vocab else '<unk>' for w in question_text]
		if question_text[-1] not in string.punctuation:
			if args.dataset_name=='marco':
				question_text.append('?')
			# else:
			nonpunc_count+=1
				# print('not ending with punctuation:text=',' '.join(question_text))
		question_text = ' '.join(question_text)
		fout_txt.write('{}\n'.format(question_text))
		fout_uid.write('{}\n'.format(uid))
	print('nonpunc_count:',nonpunc_count)
	




