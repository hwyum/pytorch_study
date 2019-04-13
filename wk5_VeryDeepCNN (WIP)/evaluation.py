import torch
from model.data import MovieDataJaso
from model.utils import JamoTokenizer
from model.network import VDCNN
import json
import gluonnlp as nlp
from tqdm import tqdm_notebook, tqdm
import fire


def eval(cfgpath, model, dev):
    """ Function for evaluation with test dataset """

    # preparation
    with open(cfgpath) as io:
        params = json.loads(io.read())

    tst_path = params['filepath'].get('tst')
    tokenizer = JamoTokenizer()
    padder = nlp.data.PadSequence(length=300)

    # Build Data Loader
    tst_dl = MovieDataJaso(tst_path, tokenizer, padder)

    # Inference
    model = model.to(dev)

    model.eval()
    correct = 0
    num_yb = 0
    for xb, yb in tqdm(tst_dl, desc='Test'):
        xb = xb.to(dev)
        yb = yb.to(dev)

        output = model(xb)
        _, predicted = torch.max(output, 1)
        correct += torch.sum((yb == predicted)).item()
        num_yb += len(num_yb)
    else:
        accuracy = correct / num_yb

    print("test accuracy : {:.3f}".format(accuracy))

if __name__ == '__main__':
    fire.Fire(eval)
