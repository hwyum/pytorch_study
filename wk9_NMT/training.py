# import os
# os.chdir('./wk9_NMT')
# import sys
# sys.path.append('/Users/haewonyum/Google 드라이브/Colab Notebooks/Pytorch_study/wk9_NMT')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from utils import prepare_sequence
from model.network import Seq2Seq
from model.data import NMTdataset
from utils import collate_fn
import fire
import json
import pickle
from pathlib import Path
from tqdm import tqdm
# from metrics import f1
import numpy as np

cfgpath = "./config.json"
dev = "cpu"



def train(cfgpath, from_checkpoint=False, model_dir="./experiments/base_model", dev=None):

    # GPU Setting
    if dev == "cpu":
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # config file parsing
    with open(Path.cwd()/cfgpath) as io:
        params = json.loads(io.read())

    # Load Vocab
    vocab_src_path = params['filepath'].get('vocab_src')
    vocab_tgt_path = params['filepath'].get('vocab_tgt')

    with open(vocab_src_path, mode='rb') as io:
        vocab_src = pickle.load(io)
    with open(vocab_tgt_path, mode='rb') as io:
        vocab_tgt = pickle.load(io)

    # Load Model
    embedding_dim = params['model'].get('embedding_dim')
    hidden_dim = params['model'].get('hidden_dim')
    model = Seq2Seq(vocab_src, vocab_tgt, embedding_dim, hidden_dim, dev)

    # Build Data Loader
    tr_path = params['filepath'].get('tr')
    val_path = params['filepath'].get('val')
    split_fn = lambda x: x.split()
    tr_ds  = NMTdataset(tr_path, vocab_src, vocab_tgt, split_fn)
    val_ds = NMTdataset(val_path, vocab_src, vocab_tgt, split_fn)
    tr_dl = DataLoader(tr_ds, batch_size=params['training'].get('batch_size'),
                       shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=params['training'].get('batch_size') * 2,
                        drop_last=False, collate_fn=collate_fn)

    # Training Parameter
    epochs = params['training'].get('epochs')
    lr = params['training'].get('learning_rate')
    opt = optim.Adam(model.parameters(), lr=lr)
    clip = params['training'].get('grad_clip')

    # if training from checkpoint
    prev_epoch = 0
    if from_checkpoint:
        model_path = params['filepath'].get('ckpt')
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['opt_state_dict'])
        for state in opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(dev)

        prev_epoch = ckpt['epoch']

    model.to(dev)

    # Training
    for epoch in tqdm(range(epochs), desc='Epoch'):
        model.train()
        tr_loss_sum = 0
        tr_nTotals = 0
        for i, mb in enumerate(tqdm(tr_dl, desc='Train Batch')):
            mb = map(lambda x: x.to(dev), mb)

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Calculate Loss.
            tr_loss, nTotal = model(mb)
            tr_loss_sum += tr_loss * nTotal
            tr_nTotals += nTotal

            # Step 3. Compute the loss, gradients, and update the parameters by calling optimizer.step()
            tr_loss.backward()

            # Clip gradients: gradients are modified in place
            _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        # eval
        else:
            tr_loss_avg = tr_loss_sum / tr_nTotals
            model.eval()

            val_loss_sum = 0
            val_nTotals = 0
            for i, mb in enumerate(tqdm(val_dl, desc='Validation Batch')):
                mb = map(lambda x: x.to(dev), mb)
                val_loss, nTotal = model(mb, use_teacher_forcing=False)
                val_loss_sum += val_loss * nTotal
                val_nTotals += nTotal

            val_loss_avg = val_loss_sum / val_nTotals
            print('Epoch: {}, training loss: {:.3f}, validation loss: {:.3f}'
                  .format(prev_epoch+epoch, tr_loss_avg, val_loss_avg))

    ckpt = {'epoch': prev_epoch+epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()}
    savepath = params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)




if __name__ == '__main__':
    fire.Fire(train)