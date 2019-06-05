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


cfgpath = './config.json'

def evaluate(model, dataloader, dev):
    """ calculate validation loss and accuracy"""
    model.eval()
    score = 0.

    for step, mb in enumerate(tqdm(dataloader, desc = 'Validation')):
        sentence, tags, length = mb
        sentence, tags = map(lambda x: x.to(dev), (sentence, tags))

        scores, tag_seqs = model(sentence)
        score += torch.mean(scores)

        # accuracy calculation
        # _correct, _num_yb = accuracy_batch(outputs, yb)
        # correct += _correct
        # num_yb += _num_yb
    else:
        score /= (step+1)
        # accuracy = correct / num_yb
    return score


def train(cfgpath):
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    model = BiLSTM_CRF(vocab, tag_to_ix, embedding_dim, hidden_dim, dev)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Build Data Loader
    tr_path = params['filepath'].get('tr')
    val_path = params['filepath'].get('val')
    tr_ds  = NER_data(tr_path, vocab, tag_to_ix)
    val_ds = NER_data(val_path, vocab, tag_to_ix)
    tr_dl = DataLoader(tr_ds, batch_size=params['training'].get('batch_size'),
                       shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=params['training'].get('batch_size') * 2,
                        drop_last=False, collate_fn=collate_fn)

    model.to(dev)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in tqdm(range(1), desc='Epoch'):  # again, normally you would NOT do 300 epochs, it is toy data

        model.train()
        for i, mb in enumerate(tqdm(tr_dl, desc='Train Batch')):
            sentence, tags, length = mb
            sentence, tags = map(lambda x: x.to(dev), (sentence, tags))

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # # Step 2. Get our inputs ready for the network, that is,
            # # turn them into Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix)
            # targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence, tags)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
            if i>3: break

        # eval
        score = evaluate(model, val_dl, dev)
        print(
            'Epoch: {}, training loss: {:.3f}, validation score: {:.3f}'
            .format(epoch, loss, score))

    # Check predictions after training
    # with torch.no_grad():
    #     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    #     print(model(precheck_sent))
    # We got it!
#
# if __name__ == '__main__':
#     fire.Fire(train)

train(cfgpath)