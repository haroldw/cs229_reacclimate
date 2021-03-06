import torch
import numpy as np
import pandas as pd
import spacy
import os
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.argmax(torch.softmax(preds,dim=1),dim=1)
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion, labelName):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    i = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        followerCnt = batch.follower_count
        labels = getattr( batch, labelName )
        predictions = model(text, text_lengths, followerCnt).squeeze(1)
        loss = criterion(predictions, labels)
        i += 1
        acc = 0
#        acc = get_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        #epoch_acc += acc.item()
        acc = 0
#    pdb.set_trace() 
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, labelName):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    i = 0
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            followerCnt = batch.follower_count
            labels = getattr( batch, labelName )
            predictions = model(text, text_lengths, followerCnt).squeeze(1)

            loss = criterion(predictions, labels)
            #print( 'Itr %s: %s' % ( i, loss.item() ) )
            i += 1
            #acc = get_accuracy(predictions, labels)
            acc = 0

            epoch_loss += loss.item()
#            epoch_acc += acc.item()
            epoch_acc += 0
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

nlp = spacy.load('en')
def predict_engagement(model, sentence, TEXT, followerCnt, device):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    preds = model(tensor, length_tensor, followerCnt)
#    preds = torch.argmax(torch.softmax(preds,dim=0))
    print( preds )
    return preds

class BatchWrapper:
    def __init__(self, dl, x_vars, y_vars):
        self.dl, self.x_vars, self.y_vars = dl, x_vars, y_vars # we pass in the list of attributes for x and y

    def __iter__(self):
        for batch in self.dl:
            if self.x_vars is not None:
                x = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.x_vars], dim=1)
            else:
                x = getattr(batch, self.x_var) # we assume only one input in this wrapper
            if self.y_vars is not None: # we will concatenate y into a single tensor
    #            y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1).long()
                y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1)
            else:
                y = torch.zeros((1))

            yield (x, y)

    def __len__(self):
        return len(self.dl)

