import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import fire
from model.data import MovieDataJaso
from model.utils import JamoTokenizer
from model.network import CharacterCNN
import json
import gluonnlp as nlp
from tqdm import tqdm_notebook, tqdm
from tensorboardX import SummaryWriter


def train(cfgpath):
    """ Training Character CNN Model with Naver Movie Dataset"""

    # config file parsing
    with open(cfgpath) as io:
        params = json.loads(io.read())

    tokenizer = JamoTokenizer()
    padder = nlp.data.PadSequence(length=300)

    # Load Model
    model = CharacterCNN(len(tokenizer.token2idx), embedding_dim=params['model'].get('embedding_dim'),
                         model_type=params['model'].get('model_type'), class_num=params['model'].get('num_classes'))  # num_embedding, embedding_dim, class_num=2

    # Build Data Loader
    tr_path = params['filepath'].get('tr')
    tst_path = params['filepath'].get('tst')
    tr_ds = MovieDataJaso(tr_path, tokenizer, padder)
    tst_ds = MovieDataJaso(tst_path, tokenizer, padder)
    tr_dl = DataLoader(tr_ds, batch_size=params['training'].get('batch_size'), shuffle=True, drop_last=True)
    tst_dl = DataLoader(tst_ds, batch_size=params['training'].get('batch_size') * 2, drop_last=False)

    # loss function and optimization
    loss_func = F.cross_entropy
    # opt = optim.SGD(model.parameters(), lr=params['training'].get('learning_rate'),
    #                 momentum=params['training'].get('momentum'))
    opt = optim.Adam(model.parameters(), lr=params['training'].get('learning_rate'))

    # Adjust learning rate (참고: torch.optim.lr_scheduler)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5, last_epoch=-1)
    epochs = params['training'].get('epochs')

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    # TensorboardX
    writer = SummaryWriter(log_dir='./runs/exp')

    # Training
    for epoch in tqdm_notebook(range(epochs), desc='Epoch'):

        model.train()
        scheduler.step()
        avg_tr_loss = 0
        # tr_step = 0

        for step, mb in enumerate(tqdm_notebook(tr_dl, desc='Training')):
            xb, yb = map(lambda x: x.to(dev), mb)
            loss, _, _ = loss_batch(model, loss_func, xb, yb, opt=opt)
            avg_tr_loss += loss
            # tr_step += 1

            if(epoch * len(tr_dl) + step) % 500 == 0:
                val_loss, _ = evaluate(model,loss_func,tst_dl,dev)
                writer.add_scalars('losses', {'tr_loss':avg_tr_loss/(step+1),
                                              'val_loss':val_loss}, epoch * len(tr_dl) + step )
                model.train()
        else:
            avg_tr_loss /= (step+1)

        model.eval()
        avg_val_loss, accuracy = evaluate(model, loss_func, tst_dl, dev)

        print('Epoch: {}, training loss: {:.3f}, test loss: {:.3f}, test accuracy: {:.3f}'
              .format(epoch, avg_tr_loss, avg_val_loss, accuracy))


    ckpt = {'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict()}
    savepath = params['filepath'].get('ckpt')
    torch.save(ckpt, savepath)


def loss_batch(model, loss_func, xb, yb, opt=None):
    correct = 0
    output = model(xb)
    loss = loss_func(output, yb)

    if opt is not None:
        loss.backward()  ## backprop
        opt.step()  ## weight update
        opt.zero_grad()  ## gradient initialize

    ## test time인 경우, accuracy 계산
    if opt is None:  # test time
        _, predicted = torch.max(output, 1)
        correct = (yb == predicted)

    return loss.item(), len(xb), correct

def evaluate(model, loss_func, dataloader, dev):
    """ calculate validation loss and accuracy"""
    model.eval()
    avg_loss = 0
    correct = 0
    num_yb = 0
    for step, mb in enumerate(tqdm_notebook(dataloader, desc = 'Validation')):
        xb, yb = map(lambda x: x.to(dev), mb)
        output = model(xb)
        loss = loss_func(output, yb)
        avg_loss += loss.item()
        # accuracy calculation
        _, predicted = torch.max(output, 1)
        # print('predicted size : ', predicted.size())
        correct += torch.sum((yb == predicted)).item()
        num_yb += len(yb)
    else:
        avg_loss /= (step+1)
        accuracy = correct / num_yb
    return avg_loss, accuracy



if __name__ == '__main__':
    fire.Fire(train)




