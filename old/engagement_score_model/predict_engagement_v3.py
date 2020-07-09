#!/usr/bin/env python
# coding: utf-8

# load the pytorch model
import torch
import random
import pandas as pd
from torchtext import data
import torch.nn as nn
import torch.optim as optim
from models import LTSM
import util
import time
import re
import pdb
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet 
from collections import defaultdict, namedtuple

SEED = 1234
MAX_VOCAB_SIZE = 10_000
BATCH_SIZE = 64 * 64
EMBEDDING_DIM = 50
HIDDEN_DIM = int(256/8)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 10
best_valid_loss = float('inf')
tPath = '../twitter/data/'
trainFile = './train.csv'
testFile = './test.csv'
valFile = './val.csv'

df = pd.read_csv(valFile)
usrGrpCnt = len(df.columns) - 1
sentCategoryCnt = len(df[df.columns[-1]].unique())
output_dim = 1

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', include_lengths = True, lower=True)
LABEL = data.LabelField(dtype = torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csvFields = [   ('text', TEXT) ]
for userGrp in range( usrGrpCnt ):
    label = 'group%s' % userGrp
    csvFields.append( ( label, LABEL ) )

train_data, valid_data, test_data = data.TabularDataset.splits(
                path='.', 
                train=trainFile,
                validation=valFile, 
                test=testFile, 
                format='csv',
                fields=csvFields,
                skip_header=True,
            )

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = "glove.twitter.27B.50d", 
                 unk_init = torch.Tensor.normal_)

INPUT_DIM = 10002
PAD_IDX = 1
modelGrp0 = LTSM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, output_dim, 
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
modelGrp1 = LTSM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, output_dim, 
            N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

model_group_zero = modelGrp0.to(device)
model_group_one = modelGrp1.to(device)

model_group_zero.load_state_dict(torch.load('lstm_model_group0.pt',map_location=device))
model_group_one.load_state_dict(torch.load('lstm_model_group1.pt',map_location=device))

# FIXME: Please update this follower count to your best estimate
wordDf = pd.read_csv('replacements.csv',encoding='ISO-8859-1')

for index, row in wordDf.iterrows():
  follower_count = torch.tensor( [[row['follower_cnt']]] ).to(device)
  refText = row['text']
  newText = re.sub( row['org'], row['replacement'], refText)
  print('Org text is %s' % refText)
  engagement = util.predict_engagement(model_group_zero, refText, TEXT, device, follower_count).item()
  print('The engagement score is %s\n' % engagement )
  print('New text is %s' % newText)
  engagement = util.predict_engagement(model_group_zero, newText, TEXT, device, follower_count).item()
  print('The engagement score is %s\n' %  engagement )