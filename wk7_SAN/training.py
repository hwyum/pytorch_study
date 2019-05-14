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


def loss_batch(model, attn_hops, loss_func, q1, q2, yb, opt, dev):
    inputs = (q1, q2)
    outputs = None
    attn_mtx_1 = None
    attn_mtx_2 = None

    for j, layer in enumerate(list(model.children())[0]):
        outputs = layer(inputs)
        inputs = outputs

        # extract attn_mtx
        if isinstance(layer, SelfAttention):
            # print('here')
            attn_mtx_1 = outputs[1][0]
            attn_mtx_2 = outputs[1][1]

    # loss with penalty term
    loss = loss_func(outputs, yb) + penalty(attn_mtx_1, attn_hops, 0.3, dev) + penalty(attn_mtx_2, attn_hops, 0.3, dev)
    loss.backward()  # backprop
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)  # gradient clipping
    opt.step()  # weight update
    opt.zero_grad()  # gradient initialize

    return loss.item(), outputs


def accuracy_batch(outputs, yb):
    # accuracy calculation
    _, predicted = torch.max(outputs, 1)
    correct = torch.sum((yb == predicted)).item()
    num_yb = len(yb)

    return correct, num_yb


def evaluate(model, loss_func, dataloader, dev):
    """ calculate validation loss and accuracy"""
    model.eval()
    avg_loss = 0
    correct = 0
    num_yb = 0
    for step, mb in enumerate(tqdm(dataloader, desc = 'Validation')):
        q1, q2, yb = map(lambda x: x.to(dev), mb)
        outputs = model((q1, q2))
        loss = loss_func(outputs, yb)
        avg_loss += loss.item()

        # accuracy calculation
        _correct, _num_yb = accuracy_batch(outputs, yb)
        correct += _correct
        num_yb += _num_yb
    else:
        avg_loss /= (step+1)
        accuracy = correct / num_yb
    return avg_loss, accuracy


def penalty(attn_mtx: torch.Tensor, hops: int, coefficient, dev):
    aat = torch.bmm(attn_mtx, attn_mtx.permute(0,2,1)).to(dev)
    identity = torch.eye(hops).to(dev)
    penalty_term = torch.norm((aat - identity), dim=(1,2)) ** 2
    penalty_term = coefficient * penalty_term.mean()

    return penalty_term


def train(cfgpath):
    """ Training SAN Model with Korean Question Pairs Dataset"""

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

    # Build Data Loader
    tr_path = params['filepath'].get('tr')
    val_path = params['filepath'].get('val')
    tr_ds = QuestionPair(tr_path, vocab, tokenizer, padder)
    val_ds = QuestionPair(val_path, vocab, tokenizer, padder)
    tr_dl = DataLoader(tr_ds, batch_size=params['training'].get('batch_size'), shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=params['training'].get('batch_size') * 2, drop_last=False)

    # loss function and optimization
    loss_func = F.cross_entropy
    opt = optim.Adam(model.parameters(), lr=params['training'].get('learning_rate'))

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    # TensorboardX
    writer = SummaryWriter(log_dir='./runs/exp')

    # Training
    epochs = params['training'].get('epochs')
    for epoch in tqdm(range(epochs), desc='Epoch'):

        model.train()
        avg_tr_loss = 0
        tr_accuracy = 0
        tr_num_yb = 0

        for step, mb in enumerate(tqdm(tr_dl, desc='Training')):
            q1, q2, yb = map(lambda x: x.to(dev), mb)
            loss, outputs = loss_batch(model, attn_hops, loss_func, q1, q2, yb, opt, dev)
            avg_tr_loss += loss

            # training accuracy
            _correct, _num_yb = accuracy_batch(outputs, yb)
            tr_accuracy += _correct
            tr_num_yb += _num_yb

            # tensorboard write
            if epoch > 0 and (epoch * len(tr_dl) + step) % 500 == 0:
                val_loss, _ = evaluate(model, loss_func, val_dl, dev)
                writer.add_scalars('losses', {'tr_loss':avg_tr_loss/(step+1),
                                              'val_loss':val_loss}, epoch * len(tr_dl) + step)
                model.train()
        else:
            avg_tr_loss /= (step+1)
            tr_accuracy /= tr_num_yb

        model.eval()
        avg_val_loss, accuracy = evaluate(model, loss_func, val_dl, dev)

        print('Epoch: {}, training loss: {:.3f}, validation loss: {:.3f}, training accuracy: {:.3f}, validation accuracy: {:.3f}'
              .format(epoch, avg_tr_loss, avg_val_loss, tr_accuracy, accuracy))

    ckpt = {'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()}
    savepath = params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)
    writer.close()


if __name__ == '__main__':
    fire.Fire(train)




