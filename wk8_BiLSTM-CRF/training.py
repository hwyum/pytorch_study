# import os
# os.chdir('./wk8_BiLSTM-CRF')
# import sys
# sys.path.append('/Users/haewonyum/Google 드라이브/Colab Notebooks/Pytorch_study/wk8_BiLSTM-CRF')

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from utils import prepare_sequence
from model.network import BiLSTM_CRF
from model.data import NER_data
from utils import collate_fn
import fire
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from metrics import f1
import numpy as np

# cfgpath = './config.json'


def evaluate(model, dataloader, dev):
    """ calculate validation score and accuracy"""
    model.eval()
    score = 0.
    f1_score = 0.

    accumulated_preds, accumulated_targets = [], []
    for step, mb in enumerate(tqdm(dataloader, desc = 'Validation')):
        sentence, tags, mask = mb
        sentence, tags, mask = map(lambda x: x.to(dev), (sentence, tags, mask))

        scores, pred_seqs = model(sentence, mask)
        score += torch.mean(scores)
        accumulated_preds.append(np.asarray(pred_seqs))
        accumulated_targets.append(tags.numpy())

    else:
        score /= (step+1)
        f1_score = f1(np.concatenate(accumulated_preds, axis=None), np.concatenate(accumulated_targets, axis=None))

    return score, f1_score


def train(cfgpath):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #dev = torch.device("cpu")

    # config file parsing
    with open(Path.cwd()/cfgpath) as io:
        params = json.loads(io.read())

    # Load Vocab
    vocab_path = params['filepath'].get('vocab')
    vocab_tag_path = params['filepath'].get('vocab_tag')
    with open(vocab_path, mode='rb') as io:
        vocab = pickle.load(io)
    with open(vocab_tag_path, mode='rb') as io:
        vocab_tag = pickle.load(io)

    # Load Model
    tag_to_ix = vocab_tag.token_to_idx
    tag_to_ix[START_TAG] = len(tag_to_ix)
    tag_to_ix[STOP_TAG] = len(tag_to_ix)
    embedding_dim = params['model'].get('embedding_dim')
    hidden_dim = params['model'].get('hidden_dim')

    model = BiLSTM_CRF(vocab, tag_to_ix, embedding_dim, hidden_dim, dev, start_tag=START_TAG, stop_tag=STOP_TAG)
    model.to(dev)

    # Build Data Loader
    tr_path = params['filepath'].get('tr')
    val_path = params['filepath'].get('val')
    tr_ds  = NER_data(tr_path, vocab, tag_to_ix)
    val_ds = NER_data(val_path, vocab, tag_to_ix)
    tr_dl = DataLoader(tr_ds, batch_size=params['training'].get('batch_size'),
                       shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=params['training'].get('batch_size') * 2,
                        drop_last=False, collate_fn=collate_fn)

    # Training Parameter
    epochs = params['training'].get('epochs')
    lr = params['training'].get('learning_rate')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in tqdm(range(epochs), desc='Epoch'):
        model.train()
        loss_avg = 0
        for i, mb in enumerate(tqdm(tr_dl, desc='Train Batch')):
            sentence, tags, mask = mb
            sentence, tags, mask = map(lambda x: x.to(dev), (sentence, tags, mask))

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Calculate NLL.
            loss = model.neg_log_likelihood(sentence, tags)
            loss_avg += torch.mean(loss)

            # Step 3. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss.backward()
            optimizer.step()

        # eval
        else:
            score, f1_score = evaluate(model, val_dl, dev)
            loss_avg /= (i+1)
            print(
                'Epoch: {}, training loss: {:.3f}, validation score: {:.3f}, validation f1 score: {:.3f}'
                .format(epoch, loss_avg, score, f1_score))


if __name__ == '__main__':
    fire.Fire(train)
