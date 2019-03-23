import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import fire
from model.network import movieCNN
from model.data_load import movie_data
from konlpy.tag import Mecab
import gluonnlp as nlp
from tqdm import tqdm, tqdm_notebook

def train(cfgpath):
    # parsing json
    with open(cfgpath) as io:
        params = json.loads(io.read())

    with open(params['filepath'].get('vocab'), mode='rb') as io:
        vocab = pickle.load(io)

    # Load Model
    model = movieCNN(vocab, class_num=params['model'].get('num_classes'))

    # Building dataset, dataLoader
    tokenizer = Mecab()
    padder = nlp.data.PadSequence(length=30)

    tr_dataset  = movie_data(params['filepath'].get('tr'), vocab, tokenizer, padder)
    tst_dataset = movie_data(params['filepath'].get('tst'), vocab, tokenizer, padder)
    tr_dl  = DataLoader(tr_dataset, batch_size=params['training'].get('batch_size'), shuffle=True, drop_last=True)
    tst_dl = DataLoader(tst_dataset, batch_size=params['training'].get('batch_size'), shuffle=True, drop_last=False)
    # Training
    loss_func = F.cross_entropy
    opt = optim.Adam(model.parameters(), lr = params['training'].get('learning_rate'))
    epochs = params['training'].get('epochs')
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    for epoch in tqdm_notebook(range(epochs), desc = 'epochs'): ## decs 가 뭐지?
        model.train()
        avg_tr_loss  = 0
        avg_tst_loss = 0
        tr_step = 0
        tst_step = 0


        for xb, yb in tqdm_notebook(tr_dl, desc='iters'):
            xb = xb.to(dev)
            yb = yb.to(dev)
            output = model(xb)

            opt.zero_grad()
            tr_loss  = loss_func(output, yb)
            reg_term = torch.norm(model.linear.weight, p=2)
            tr_loss.add_(0.5 * reg_term)
            tr_loss.backward() # backprop
            opt.step() # weight update

            avg_tr_loss = tr_loss.item()
            tr_step += 1
        else: ## 이런 문법은 처음 본다...
            avg_tr_loss /= tr_step

        model.eval()
        correct = 0
        total = 0
        for xb, yb in tqdm_notebook(tst_dl, desc='iters'):
            xb = xb.to(dev)
            yb = yb.to(dev)

            with torch.no_grad():
                output = model(xb)
                tst_loss = loss_func(output, yb)
                avg_tst_loss += tst_loss.item()
                tst_step += 1

                # Accuracy 계산
                _, predicted = torch.max(output, 1)
#                 print('output: ', output.size, 'predicted: ', predicted.size, 'yb: ', yb.size)
                total += yb.size(0)
                correct += (yb == predicted).sum().item()

        else:
            avg_tst_loss /= tst_step
            accuracy = correct / total

        tqdm.write('epoch : {}, tr_loss : {:3f}, tst_loss : {:3f}, tst_acc : {:2f}'.format(epoch+1, avg_tr_loss, avg_tst_loss, accuracy))

    ckpt = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'vocab': vocab}

    savepath = params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)


if __name__ == '__main__':
    fire.Fire(train)

