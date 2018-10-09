import json
import spacy
import argparse
import re
import tqdm
import unicodedata
import pdb
from collections import Counter

test_mode=True
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str, default='../data/', help='dataset folder.')
parser.add_argument('--dataset_name',type=str, default='squad')
parser.add_argument('--output_path', type=str, default='data/')
args = parser.parse_args()

def normalize_text(text):
	return unicodedata.normalize('NFD', text)
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
	questions_tokened = nlp.pipe(questions, batch_size=10000, n_threads=32)
	for tokened in tqdm.tqdm(questions_tokened, total=len(questions)):
		word_counter.update([normalize_text(w.text) for w in tokened if len(normalize_text(w.text)) > 0])
vocab = sorted([w for w in word_counter], key=word_counter.get, reverse=True)
# print('size of vocab:',len(vocab))
# print('appear more than 10 times:', len([w for w in word_counter if word_counter.get(w)>=10]))
# pdb.set_trace()
vocab=vocab[:10000]
        

for part_name,out_name in zip(process_orders,output_names):
	questions,uids = load_data(args.data_path+args.dataset_name+'/{}.json'.format(part_name))
	questions = [reform_text(question) for question in questions]
	questions_tokened = nlp.pipe(questions, batch_size=10000, n_threads=32)
	add_str='_test' if test_mode else ''
	fout_txt=open(args.output_path+args.dataset_name+add_str+'/{}.txt'.format(out_name),'w')
	fout_uid=open(args.output_path+args.dataset_name+add_str+'/{}_id.txt'.format(out_name),'w')
	for tokened,uid in tqdm.tqdm(zip(questions_tokened, uids), total=len(questions)):
		question_text = [normalize_text(w.text) for w in tokened if len(normalize_text(w.text)) > 0]
		question_text = ' '.join([w if w in vocab else '<unk>' for w in question_text])
		fout_txt.write('{}\n'.format(question_text))
		fout_uid.write('{}\n'.format(uid))
			
	




