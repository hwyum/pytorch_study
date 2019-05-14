# import sys
# sys.path.append('/Users/haewonyum/Google 드라이브/Colab Notebooks/Pytorch_study/wk7_SAN')
#
# import os
# os.chdir('/Users/haewonyum/Google 드라이브/Colab Notebooks/Pytorch_study/wk7_SAN')

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import fire
from model.data import QuestionPair
from model.network import SAN
from model.modules import SelfAttention
import json
import gluonnlp as nlp
from tqdm import tqdm_notebook, tqdm
from tensorboardX import SummaryWriter
from konlpy.tag import Mecab
import pickle

# cfgpath = 'config.json'


def loss_batch(model, attn_hops, loss_func, q1, q2, y, opt):
    inputs = (q1, q2)
    outputs = None
    attn_mtx_1 = None
    attn_mtx_2 = None

    for j, layer in enumerate(list(model.children())[0]):
        outputs = layer(inputs)
        inputs = outputs

        # print(j, type(layer))
        if isinstance(layer, SelfAttention):
            # print('here')
            attn_mtx_1 = outputs[1][0]
            attn_mtx_2 = outputs[1][1]

    loss = loss_func(outputs, y) + penalty(attn_mtx_1, attn_hops, 0.3) + penalty(attn_mtx_2, attn_hops, 0.3)
    loss.backward()  ## backprop
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)  # gradient clipping
    opt.step()  ## weight update
    opt.zero_grad()  ## gradient initialize

    return loss.item()


def evaluate(model, loss_func, dataloader, dev):
    """ calculate validation loss and accuracy"""
    model.eval()
    avg_loss = 0
    correct = 0
    num_yb = 0
    for step, mb in enumerate(tqdm(dataloader, desc = 'Validation')):
        q1, q2, y = map(lambda x: x.to(dev), mb)
        output = model((q1, q2))
        loss = loss_func(output, y)
        avg_loss += loss.item()

        # accuracy calculation
        _, predicted = torch.max(output, 1)
        # print('predicted size : ', predicted.size())
        correct += torch.sum((y == predicted)).item()
        num_yb += len(y)
    else:
        avg_loss /= (step+1)
        accuracy = correct / num_yb
    return avg_loss, accuracy


def penalty(attn_mtx: torch.Tensor, hops: int, coefficient):
    aat = torch.bmm(attn_mtx, attn_mtx.permute(0,2,1))
    identity = torch.eye(hops)
    penalty_term = torch.norm((aat - identity), dim=(1,2)) ** 2
    penalty_term = coefficient * penalty_term.mean()

    return penalty_term


def train(cfgpath):
    """ Training ConvRec Model with Naver Movie Dataset"""

    # config file parsing
    with open(cfgpath) as io:
        params = json.loads(io.read())

    # load Vocab
    with open(params['filepath'].get('vocab'), mode='rb') as io:
        vocab = pickle.load(io)

    tokenizer = Mecab().morphs
    padder = nlp.data.PadSequence(length=56)


    # Load Model
    num_embedding = len(vocab)
    embedding_dim = params['model'].get('embedding_dim')
    lstm_hidden = params['model'].get('lstm_hidden')
    attn_hidden = params['model'].get('attn_hidden')
    attn_hops = params['model'].get('attn_hops')
    class_num = params['model'].get('num_classes')
    fc_hidden = params['model'].get('fc_hidden')
    model = SAN(num_embedding, embedding_dim, lstm_hidden, attn_hidden, attn_hops, fc_hidden, class_num)


    # print(model)

    # Build Data Loader
    tr_path = params['filepath'].get('tr')
    val_path = params['filepath'].get('val')
    tr_ds = QuestionPair(tr_path, vocab, tokenizer, padder)
    val_ds = QuestionPair(val_path, vocab, tokenizer, padder)
    tr_dl = DataLoader(tr_ds, batch_size=params['training'].get('batch_size'), shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=params['training'].get('batch_size') * 2, drop_last=False)

    # # ----------- test --------------- #
    #
    # for i, inputs in enumerate(tr_dl):
    #     if i > 0: break
    #     q1, q2, label = inputs
    #     inputs = (q1, q2)
    #     for j, layer in enumerate(list(model.children())[0]):
    #         outputs = layer(inputs)
    #         inputs = outputs
    #         print(j, type(layer))
    #         if isinstance(layer, SelfAttention):
    #             print('here')
    #             print(outputs[1][0].shape, outputs[1][1].shape)
    #             attn_mat_1 = outputs[1][0]
    #             attn_mat_2 = outputs[1][1]
    #
    #     # output = model((q1, q2))
    #
    #
    # print(outputs.size())    # batch x 2
    #
    # exit(-1)
    #
    # # -------------------------------- #

    # loss function and optimization
    loss_func = F.cross_entropy
    opt = optim.Adam(model.parameters(), lr=params['training'].get('learning_rate'))
#     opt = optim.Adadelta(model.parameters(), lr=params['training'].get('learning_rate'), rho=0.95, eps=1e-5)

    # Adjust learning rate (참고: torch.optim.lr_scheduler)
#     scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5, last_epoch=-1)
    epochs = params['training'].get('epochs')

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    # TensorboardX
    writer = SummaryWriter(log_dir='./runs/exp')

    # Training
    for epoch in tqdm(range(epochs), desc='Epoch'):

        model.train()
#         scheduler.step()
        avg_tr_loss = 0

        for step, mb in enumerate(tqdm(tr_dl, desc='Training')):
            q1, q2, y = map(lambda x: x.to(dev), mb)
            loss = loss_batch(model, attn_hops, loss_func, q1, q2, y, opt)
            avg_tr_loss += loss

            if epoch > 0 and (epoch * len(tr_dl) + step) % 500 == 0:
                val_loss, _ = evaluate(model, loss_func, val_dl, dev)
                writer.add_scalars('losses', {'tr_loss':avg_tr_loss/(step+1),
                                              'val_loss':val_loss}, epoch * len(tr_dl) + step )
                model.train()
        else:
            avg_tr_loss /= (step+1)

        model.eval()
        avg_val_loss, accuracy = evaluate(model, loss_func, val_dl, dev)

        print('Epoch: {}, training loss: {:.3f}, validation loss: {:.3f}, validation accuracy: {:.3f}'
              .format(epoch, avg_tr_loss, avg_val_loss, accuracy))

    ckpt = {'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()}
    savepath = params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)
    writer.close()








# train('./config.json')

if __name__ == '__main__':
    fire.Fire(train)




