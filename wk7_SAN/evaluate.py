import torch
from torch.utils.data import DataLoader
import fire
from model.data import QuestionPair
from model.network import SAN
import json
import gluonnlp as nlp
from tqdm import tqdm_notebook, tqdm
from konlpy.tag import Mecab
import pickle


def _accuracy_batch(outputs, yb):
    # accuracy calculation
    _, predicted = torch.max(outputs, 1)
    correct = torch.sum((yb == predicted)).item()
    num_yb = len(yb)

    return correct, num_yb


def get_accuracy(model, dataloader, dev):
    """ calculate test accuracy"""
    model.eval()
    avg_loss = 0
    correct = 0
    num_yb = 0
    for step, mb in enumerate(tqdm(dataloader, desc = 'Test')):
        q1, q2, yb = map(lambda x: x.to(dev), mb)
        outputs = model((q1, q2))

        # accuracy calculation
        _correct, _num_yb = _accuracy_batch(outputs, yb)
        correct += _correct
        num_yb += _num_yb
    else:
        accuracy = correct / num_yb
    return accuracy


def evaluate(cfgpath):
    """ Evaluate SAN Model with Korean Question Pair Dataset (test dataset) """

    # config file parsing
    with open(cfgpath) as io:
        params = json.loads(io.read())

    # load Vocab
    with open(params['filepath'].get('vocab'), mode='rb') as io:
        vocab = pickle.load(io)

    tokenizer = Mecab().morphs
    padder = nlp.data.PadSequence(length=56)

    # Load Model
    model_path = params['filepath'].get('ckpt')
    ckpt = torch.load(model_path)

    num_embedding = len(vocab)
    embedding_dim = params['model'].get('embedding_dim')
    lstm_hidden = params['model'].get('lstm_hidden')
    attn_hidden = params['model'].get('attn_hidden')
    attn_hops = params['model'].get('attn_hops')
    class_num = params['model'].get('num_classes')
    fc_hidden = params['model'].get('fc_hidden')

    model = SAN(num_embedding, embedding_dim, lstm_hidden, attn_hidden, attn_hops, fc_hidden, class_num)
    model.load_state_dict(ckpt['model_state_dict'])

    # Build Data Loader
    tst_path = params['filepath'].get('tst')
    tst_ds = QuestionPair(tst_path, vocab, tokenizer, padder)
    tst_dl = DataLoader(tst_ds, batch_size=params['training'].get('batch_size') * 2, drop_last=False)

    # GPU Setting
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)

    model.eval()
    accuracy = get_accuracy(model, tst_dl, dev)

    print("test dataset accuract: {:.3f}".format(accuracy))


if __name__ == '__main__':
    fire.Fire(evaluate)
