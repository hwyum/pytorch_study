import argparse
from path import Path
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from konlpy.tag import Mecab
from tqdm import tqdm, tqdm_notebook
import gluonnlp as nlp
from model.data import SentencePair, collate_fn
from model.network import StochasticAnswerNetwork
from tokenizer import JamoTokenizer
from utils import Config, CheckpointManager, SummaryManager


data_dir = './data'
model_dir = './experiments/base_model'
data_dir = Path(data_dir)
model_dir = Path(model_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=data_dir, help='directory containing data and data configuration in json format')
parser.add_argument('--model_dir', default=model_dir, help='directory containing model configuration in json format')

def evaluate(model, loss_fn, val_dl, dev):
    avg_val_loss = 0
    val_acc = 0
    val_num_y = 0

    model.eval()
    for step, mb in enumerate(tqdm(val_dl, desc='evaluation')):
        y = mb[-1]
        mb = map(lambda x: x.to(dev), mb)

        scores = model(mb)
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

# data_config = Config(data_dir / 'config.json')
# model_config = Config(model_dir / 'config.json')

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    data_config = Config(data_dir / 'config.json')
    model_config = Config(model_dir / 'config.json')

    # Vocab and Tokenizer
    with open(data_config.word_vocab_path, mode='rb') as io:
        word_vocab = pickle.load(io)
    word_tokenizer = Mecab().morphs

    with open(data_config.char_vocab_path, mode='rb') as io:
        char_vocab = pickle.load(io)
    char_tokenizer = JamoTokenizer().tokenize
    char_padder = nlp.data.PadSequence(length=model_config.char_max_len)

    # Model
    model = StochasticAnswerNetwork(word_vocab, char_vocab, model_config.word_embedding_dim, model_config.char_embedding_dim,
                                    model_config.embedding_output_dim, model_config.hidden_size)

    # DataLoader
    tr_ds = SentencePair(data_config.tr_path, word_vocab, char_vocab, word_tokenizer, char_tokenizer, char_padder)
    val_ds = SentencePair(data_config.val_path, word_vocab, char_vocab, word_tokenizer, char_tokenizer, char_padder)

    tr_dl = DataLoader(tr_ds, batch_size=model_config.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dl = DataLoader(tr_ds, batch_size=model_config.batch_size * 2, collate_fn=collate_fn, shuffle=False)

    # loss function and optimization
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=model_config.learning_rate)

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    # Experiments Management
    writer = SummaryWriter("{}/runs".format(model_dir))
    checkpoint_manager = CheckpointManager(model_dir)
    summary_manager = SummaryManager(model_dir)
    best_val_loss = 1e10

    # Training
    epochs = model_config.epochs
    for epoch in tqdm(range(epochs), desc='Epoch'):

        avg_tr_loss = 0
        tr_acc = 0
        tr_num_y = 0

        model.train()
        for step, mb in enumerate(tqdm(tr_dl, desc='training')):
            y = mb[-1]
            mb = map(lambda x: x.to(dev), mb)

            opt.zero_grad()
            scores = model(mb)
            loss = loss_fn(scores, y)
            loss.backward()
            opt.step()

            avg_tr_loss += loss.item()

            # training accuracy
            _correct, _num_y = accuracy_batch(scores, y)
            tr_acc += _correct
            tr_num_y += _num_y

            if (epoch * len(tr_dl) + step) % model_config.summary_step == 0:
                avg_val_loss, val_acc = evaluate(model, loss_fn, val_dl, dev)
                writer.add_scalars(
                    "loss",
                    {"train": avg_tr_loss / (step+1), "val": avg_val_loss},
                    epoch * len(tr_dl) + step
                )
            model.train()
        else:
            avg_tr_loss /= (step + 1)
            tr_acc /= tr_num_y

            avg_val_loss, val_acc = evaluate(model, loss_fn, val_dl, dev)

            tr_summary = {"loss": avg_tr_loss, "acc": tr_acc}
            val_summary = {"loss": avg_val_loss, "acc": val_acc}

            tqdm.write(
                "epoch: {}, tr_loss: {:.3f}, val_loss: {:.3f}, tr_acc: {:.3f}, val_acc: {:.3f}".format(
                    epoch + 1,
                    tr_summary["loss"],
                    val_summary["loss"],
                    tr_summary["acc"],
                    val_summary["acc"]
                )
            )

            is_best  = avg_val_loss < best_val_loss

            if is_best:
                state = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "opt_state_dict": opt.state_dict()
                }
                summary = {"train": tr_summary, "validation": val_summary}

                summary_manager.update(summary)
                summary_manager.save("summary.json")
                checkpoint_manager.save_checkpoint(state, "best.tar")

                best_val_loss = avg_val_loss
