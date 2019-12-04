import argparse
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from konlpy.tag import Mecab
from tqdm import tqdm, tqdm_notebook
from model.network import SentenceCNN
from model.data import Corpus
from model.utils import Tokenizer, PadSequence
from utils import Config, SummaryManager, CheckpointManager
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help='directory containing data and data configuration in json format')
parser.add_argument('--model_dir', default='./experiments/base_model', help='directory containing model configuration in json format')

def evaluate(model, loss_fn, val_dl, dev):
    """ calculate validation loss and accuracy """
    model.eval()
    avg_loss = 0
    correct = 0
    num_yb = 0
    for step, mb in enumerate(tqdm(val_dl, desc='Validation')):
        xb, yb = map(lambda x: x.to(dev), mb)
        output = model(xb)
        loss = loss_fn(output, yb)
        avg_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += torch.sum((yb==predicted)).item()
        num_yb += len(yb)
    else:
        avg_loss /= (step+1)
        accuracy = correct / num_yb
    return avg_loss, accuracy


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    data_config = Config(data_dir / 'config.json')
    model_config = Config(model_dir / 'config.json')

    # Vocab and Tokenizer
    with open(data_config.vocab_path, mode='rb') as io:
        vocab = pickle.load(io)

    tokenizer = Tokenizer(vocab=vocab, split_fn=Mecab().morphs)
    padder = PadSequence(length=30, pad_val=vocab.token_to_idx['<pad>'])

    # Load Model
    model = SentenceCNN(vocab, model_config.num_classes)

    # Data Loader
    tr_dataset = Corpus(data_config.tr_path, vocab, tokenizer, padder, sep='\t')
    val_dataset = Corpus(data_config.val_path, vocab, tokenizer, padder, sep='\t')
    tr_dl  = DataLoader(tr_dataset, model_config.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_dataset, model_config.batch_size * 2, shuffle=False, drop_last=False)

    # Loss function and Optimization
    loss_func = F.cross_entropy
    opt = optim.Adam(model.parameters(), lr = model_config.learning_rate)
    epochs = model_config.epochs

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    # Experiments Management
    writer = SummaryWriter("{}/runs".format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e10

    for epoch in tqdm(range(epochs), desc = 'epochs'):
        model.train()
        avg_tr_loss  = 0
        avg_tst_loss = 0
        correct = 0
        num_y = 0
        tr_step = 0
        tst_step = 0

        for xb, yb in tqdm(tr_dl, desc='iters'):
            xb = xb.to(dev)
            yb = yb.to(dev)
            scores = model(xb)

            opt.zero_grad()
            tr_loss  = loss_func(scores, yb)
            reg_term = torch.norm(model.linear.weight, p=2)
            tr_loss.add_(0.5 * reg_term)
            tr_loss.backward() # backprop
            opt.step() # weight update

            avg_tr_loss += tr_loss.item()
            tr_step += 1

            # accuracy
            _, pred = torch.max(scores, 1)
            correct += torch.sum((pred == yb)).item()
            num_y += len(yb)

        else: ## 이런 문법은 처음 본다...
            avg_tr_loss /= tr_step
            tr_acc = correct / num_y

            # evaluation
            avg_dev_loss, dev_acc = evaluate(model, loss_func, val_dl, dev)

            # Summary
            tr_summary = {"loss": avg_tr_loss, "acc": tr_acc}
            val_summary = {"loss": avg_dev_loss, "acc": dev_acc}
            tqdm.write('epoch : {}, tr_loss : {:3f}, tst_loss : {:3f}, tr_acc: {:2f}, tst_acc : {:2f}'
                       .format(epoch+1, tr_summary["loss"], val_summary["loss"], tr_summary["acc"], val_summary["acc"]))

            is_best = avg_dev_loss < best_val_loss

            if is_best:
                state = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'opt_state_dict': opt.state_dict(),
                        'vocab': vocab
                        }
                summary = {"train": tr_summary, "validation": val_summary}

                summary_manager.update(summary)
                summary_manager.save("summary.json")
                checkpoint_manager.save_checkpoint(state, "best.tar")

                best_val_loss = avg_dev_loss
