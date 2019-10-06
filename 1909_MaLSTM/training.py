import argparse
from path import Path
from utils import Config
from model.data import SentencePair, BucketedSampler, collate_fn, BucketBatchSampler
from model.network import MaLSTM
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from konlpy.tag import Mecab
from tqdm import tqdm, tqdm_notebook

data_dir = './data'
model_dir = './experiments/base_model'
data_dir = Path(data_dir)
model_dir = Path(model_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help='directory containing data and data configuration in json format')
parser.add_argument('--model_dir', default='./experiments/base_model', help='directory containing model configuration in json format')

def evaluate(model, loss_fn, val_dl, dev):
    avg_val_loss = 0
    val_acc = 0
    val_num_y = 0

    model.eval()
    for step, mb in enumerate(tqdm(val_dl, desc='evaluation')):
        sen1, sen2, y = map(lambda x: x.to(dev), mb)
        scores = model((sen1, sen2))
        loss = loss_fn(scores, y)

        avg_val_loss += loss.item()
        correct, num_y = accuracy_batch(scores, y)
        val_acc += correct
        val_num_y += num_y

    else:
        avg_val_loss /= (step + 1)
        val_acc = val_acc / val_num_y

    return avg_val_loss, val_acc



def accuracy_batch(scores, yb):
    # accuracy calculation
    _, predicted = torch.max(scores, 1)
    correct = torch.sum((yb == predicted)).item()
    num_yb = len(yb)

    return correct, num_yb

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    data_config = Config(data_dir / 'config.json')
    model_config = Config(model_dir / 'config.json')

    # Vocab and Tokenizer
    with open(data_config.vocab_path, mode='rb') as io:
        vocab = pickle.load(io)
    tokenizer = Mecab().morphs

    # Model
    model = MaLSTM(vocab, model_config.embedding_dim, model_config.hidden_size)

    # DataLoader
    tr_ds  = SentencePair(data_config.tr_path, vocab, tokenizer)
    val_ds = SentencePair(data_config.val_path, vocab, tokenizer)

    batch_sampler = BucketBatchSampler(tr_ds, model_config.batch_size, drop_last=True, sort_key=lambda x:len(x[0]),
                                       bucket_size_multiplier=model_config.batch_size_multiplier)
    tr_dl = DataLoader(tr_ds, batch_sampler=batch_sampler, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=model_config.batch_size * 2, shuffle=True, collate_fn=collate_fn, drop_last=False)

    # loss function and optimization
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=model_config.learning_rate)

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    #todo: TensorboardX
    pass

    # Training
    epochs = model_config.epochs
    for epoch in tqdm(range(epochs), desc='Epoch'):


        avg_tr_loss = 0
        tr_acc = 0
        tr_num_y = 0

        model.train()
        for step, mb in enumerate(tqdm(tr_dl, desc='training')):
            sen1, sen2, y = map(lambda x: x.to(dev), mb)

            opt.zero_grad()
            scores = model((sen1, sen2))
            loss = loss_fn(scores, y)
            loss.backward()
            opt.step()

            avg_tr_loss += loss.item()

            # training accuracy
            _correct, _num_y = accuracy_batch(scores, y)
            tr_acc += _correct
            tr_num_y += _num_y

        else:
            avg_tr_loss /= (step + 1)
            tr_acc /= tr_num_y

        avg_val_loss, val_acc = evaluate(model, loss_fn, val_dl, dev)
        print("epoch: {}, tr_loss: {:.3f}, tr_acc: {:.3f}, val_loss: {:.3f}, val_acc: {:.3f}".
              format(epoch+1, avg_tr_loss, tr_acc, avg_val_loss, val_acc))

