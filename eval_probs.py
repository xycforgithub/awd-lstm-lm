import argparse
import time
import pdb
import math
import numpy as np
np.random.seed(331)
import torch
import torch.nn as nn
from collections import defaultdict
import json

import data
import model

from utils import batchify, get_batch, repackage_hidden, get_ids

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/marco',
                    help='location of the data corpus')
parser.add_argument('--original_data', type=str, default='data/squad',
                    help='location of the original training data corpus')
parser.add_argument('--burn_in', type=int, default=5, help='run this amount before start evaluation.')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--save_name', type=str, default='m_m', help='string for saving. originalData_evaluateData')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.original_data.encode()).hexdigest())
# print(args.original_data)
# input(fn)
if os.path.exists(fn):
    print('Loading cached original dataset...')
    corpus = torch.load(fn)
    dictionary = corpus.dictionary
else:
    input('Producing a new dataset. Maybe something is wrong?')
    corpus = data.Corpus(args.original_data)
    torch.save(corpus, fn)
    dictionary = corpus.dictionary

if '<unk>' not in dictionary.word2idx:
    print('using temp fix: <unk> -> 0')
    dictionary.word2idx['<unk>']=0

corpus = data.RefinedCorpus(args.data, dictionary, args.batch_size, args)


eval_batch_size = 10
test_batch_size = 1
train_data, train_uids = corpus.train, corpus.train_uids
val_data, valid_uids = corpus.valid, corpus.valid_uids
test_data, test_uids = corpus.test, corpus.test_uids
# pdb.set_trace()

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)

criterion = nn.CrossEntropyLoss(reduction='none')

###############################################################################
# Training code
###############################################################################

def evaluate_sents(data_source, uids, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    if args.model == 'QRNN': model.reset()
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    sent_loss = defaultdict(list)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data_batch = torch.load(open('test.pickle','rb'))
        data_test=data_batch['data']
        targets_test = data_batch['targets']
        data, targets = get_batch(data_source, i, args, evaluation=True)
        batch_uids = get_ids(uids, i, args, evaluation=True)
        # pdb.set_trace()
        output, hidden = model(data, hidden, decode=True)
        output_flat = output.view(-1, ntokens)
        per_word_loss = criterion(output_flat, targets)
        batch_uids_list = batch_uids.reshape(-1).tolist()
        loss_list = per_word_loss.tolist()
        for loss, uid in zip(loss_list, batch_uids_list):
            sent_loss[uid].append(loss)
        incre = (torch.mean(per_word_loss).item()*len(data))
        total_loss += incre
        # print('incre=',incre)
        hidden = repackage_hidden(hidden)
        # pdb.set_trace()
    avg_sent_loss = {}
    for (uid, losses) in sent_loss.items():
        avg_sent_loss[uid]=float(np.mean(losses))
    # pdb.set_trace()
    return total_loss / len(data_source), avg_sent_loss



# Load the best saved model.
with open(args.save, 'rb') as f:
    try:
        model, _, _ = torch.load(f)
    except:
        model = torch.load(f)
        print('loaded without loss and opt.')


# Loop over epochs.
lr = args.lr
# eval_loss, avg_sent_loss = evaluate_sents(val_data, valid_uids, batch_size = 10)
# print('valid loss {:5.2f} | valid ppl {:8.2f}'.format(eval_loss, math.exp(eval_loss)))

train_loss, avg_sent_loss = evaluate_sents(train_data, train_uids, batch_size = 10)
print('train loss {:5.2f} | train ppl {:8.2f}'.format(train_loss, math.exp(train_loss)))


json.dump(avg_sent_loss,open('lm_scores_{}.json'.format(args.save_name),'w'))
for uid, sent in corpus.train_dict.items():
    if 'burnin' in uid:
        continue
    print('{}\t{}'.format(sent, avg_sent_loss[uid]))
    pdb.set_trace()





