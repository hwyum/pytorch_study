import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
from pathlib import Path
from utils import Config
import pickle
from model.network import SentenceCNN
from model.data import Corpus
from model.utils import Tokenizer, PadSequence
from utils import CheckpointManager, SummaryManager
from konlpy.tag import Mecab
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data', help='directory containing test dataset')
parser.add_argument('--model_dir', default='./experiments/base_model', help='directory containing model parameters')


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

    # load vocab
    with open(data_config.vocab_path, mode='rb') as io:
        vocab = pickle.load(io)

    tokenizer = Tokenizer(vocab=vocab, split_fn=Mecab().morphs)
    padder = PadSequence(length=30, pad_val=vocab.token_to_idx['<pad>'])

    # Load Model
    checkpoint_manager = CheckpointManager(model_dir)
    state = checkpoint_manager.load_checkpoint('best.tar')
    model = SentenceCNN(vocab, model_config.num_classes)
    model.load_state_dict(state["model_state_dict"])

    # Build Data Loader
    tst_ds = Corpus(data_config.tst_path, vocab, tokenizer, padder, sep='\t')
    tst_dl = DataLoader(tst_ds, batch_size=model_config.batch_size * 2, drop_last=False)

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    # get accuracy and loss
    tst_loss, tst_acc = evaluate(model, F.cross_entropy, tst_dl, dev)
    print("test accuracy: {:.3f}, test loss: {:.3f}".format(tst_acc, tst_loss))

    # # summary update
    # test_summary = {'acc': tst_acc, 'loss': tst_loss}
    # summary_manager = SummaryManager.load('summary.json')
    # summary = summary_manager._summary
    # summary['test'] = test_summary
    # summary_manager.update(summary)
    # summary_manager.save("summary.json")

