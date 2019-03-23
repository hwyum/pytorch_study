import torch
import torch.nn as nn
import torch.functional as F
from torch import optim
from torch.utils.data import DataLoader
import fire
from model.data import MovieDataJaso
from model.utils import JamoTokenizer
from model.network import CharacterCNN
import json
import os
import gluonnlp as nlp


def train(cfgpath):
    """ Training Character CNN Model with Naver Movie Dataset"""

    ## config file parsing
    with os.open(cfgpath) as io:
        params = json.load(io.read())

    tokenizer = JamoTokenizer()
    padder = nlp.data.PadSequence(length=300)

    ## Load Model
    model = CharacterCNN(len(tokenizer.token2idx), embedding_dim=params['model'].get('embedding_dim'),
                           class_num=params['model'].get('class_num')) # num_embedding, embedding_dim, class_num=2

    ## Build Data Loader
    tr_path = params['filepath'].get('tr')
    tst_path = params['filepath'].get('tst')
    tr_ds = MovieDataJaso(tr_path, tokenizer, padder)
    tst_ds = MovieDataJaso(tst_path, tokenizer, padder)
    tr_dl = DataLoader(tr_ds, batch_size=params['training'].get('batch_size'), shuffle=True, drop_last=True)
    tst_dl = DataLoader(tst_ds, batch_size=params['training'].get('batch_size')*2, drop_last=True)

    ## loss function and optimization
    loss_func = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=params['training'].get('learning_rate'),
                    momentum=params['training'].get('momentum'))
    # Adjust learning rate (참고: torch.optim.lr_scheduler)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5, last_epoch=-1)
    eopchs = params['training'].get('eopchs')

    ## GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)


if __name__ == '__main__':
    fire.Fire(train)