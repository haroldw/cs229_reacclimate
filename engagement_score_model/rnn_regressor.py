import torch
import random
import pandas as pd
import numpy as np
from torchtext import data
import torch.nn as nn
import torch.optim as optim
from models import LTSM
import util
import time
import pdb


######################################################
#Hyperparameters and config variables
######################################################
SEED = 1234
MAX_VOCAB_SIZE = 10_000
BATCH_SIZE = 64 * 64
EMBEDDING_DIM = 50
HIDDEN_DIM = int(256/8)
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 10

for tarGrp in [ 0, 1 ]:
    best_valid_loss = float('inf')

    trainFile = './trainGrp%s.csv' % tarGrp
    valFile = './valGrp%s.csv' % tarGrp
    testFile = './testGrp%s.csv' % tarGrp

    df = pd.read_csv(valFile)
    df = df[ df['group']==tarGrp ]
    #usrGrpCnt = len(df.columns) - 1
    usrGrpCnt = 2
    #OUTPUT_DIM = len(df[df.columns[-1]].unique())
    #OUTPUT_DIM = len(df['engagement'].unique())
    OUTPUT_DIM = 1
    labelName = 'group%s' % tarGrp
    modelName = 'lstm_model_%s.pt' % labelName
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field( tokenize='spacy',
                       include_lengths=True,
                       lower=True,
                     )
    FCNT = data.Field( sequential=False,
                       use_vocab=False,
                       dtype=torch.float,
                     )
    LABEL = data.LabelField(dtype = torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    csvFields = [ ('text', TEXT),
                  ('usrGrp', None),
                  ('retweet_count', None),
                  ('favorite_count', None),
                  ('follower_count', FCNT),
                  (labelName, LABEL),
                ]

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
                     vectors = "glove.twitter.27B.%sd" % EMBEDDING_DIM,
                     unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch = True,
        device = device)

    INPUT_DIM = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = LTSM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
                N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters())
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss( reduction='mean' )

    model = model.to(device)
    criterion = criterion.to(device)

    print('The model has %s trainable parameters' % 
            util.count_parameters(model))
    print(pretrained_embeddings.shape)
    print(model.embedding.weight.data)
   
    print( 'Just started:' )
    valid_loss, valid_acc = util.evaluate(model, valid_iterator, criterion, labelName)
    print(f'\t Val. Loss: {valid_loss:.3e} |  Val. Acc: {valid_acc*100:.2e}%')
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = util.train(model, train_iterator, 
                optimizer, criterion, labelName)
        valid_loss, valid_acc = util.evaluate(model, valid_iterator,
                criterion, labelName)
        
        end_time = time.time()
        epoch_mins, epoch_secs = util.epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), modelName )
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3e} | Train Acc: {train_acc*100:.2e}%')
        print(f'\t Val. Loss: {valid_loss:.3e} |  Val. Acc: {valid_acc*100:.2e}%')
    model.load_state_dict(torch.load( modelName ))
    test_loss, test_acc = util.evaluate(model, test_iterator, criterion, labelName)
    print(f'Test Loss: {test_loss:.3e} | Test Acc: {test_acc*100:.2e}%')

    followerCnt = torch.tensor( [[0.2]] ).to(device)
    util.predict_engagement(model, 'Climate change is terrible', TEXT, followerCnt, device)
    util.predict_engagement(model, 'We need to act now to fix climate change', TEXT, followerCnt, device)
