import random
import math
import torch
import pandas as pd
import gluonnlp as nlp
from konlpy.tag import Mecab
from typing import Tuple, List
from torch.utils.data import Sampler, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from typing import Callable, Iterable


class SentencePair(Dataset):
    """ Korean Question Pair Dataset """
    def __init__(self, datapath:str, vocab:nlp.Vocab, tokenizer) -> None:
        self.df = pd.read_csv(datapath)
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.question1_idx = self.df.question1.map(lambda x: [vocab.token_to_idx[token] for token in tokenizer(x)])
        self.question2_idx = self.df.question2.map(lambda x: [vocab.token_to_idx[token] for token in tokenizer(x)])

    def __len__(self) -> int :
        """ return dataset length """
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1 = self.df.iloc[idx].question1
        q2 = self.df.iloc[idx].question2
        label = self.df.iloc[idx].is_duplicate

        q1_tokenized_toidx = [self.vocab.token_to_idx[token] for token in self.tokenizer(q1)]
        q2_tokenized_toidx = [self.vocab.token_to_idx[token] for token in self.tokenizer(q2)]

        sample = (torch.tensor(q1_tokenized_toidx), torch.tensor(q2_tokenized_toidx), torch.tensor(label))

        return sample


class SortedSampler(Sampler):
    """ Sampling from sorted data """
    def __init__(self, data_source:Iterable, sort_key:Callable):
        super().__init__(data_source)
        self.data_source = data_source
        self.sort_key = sort_key
        zip = [(i, self.sort_key(row)) for i, row in enumerate(self.data_source)]
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data_source)


class ShuffleBatchSampler(BatchSampler):
    def __init__(
            self,
            sampler,
            batch_size,
            drop_last,
            shuffle=True,
    ):
        self.shuffle = shuffle
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        # NOTE: This is not data
        batches = list(super().__iter__())
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)


def _identity(e):
    return e

class BucketBatchSampler(object):
    """ Batches are sampled from sorted buckets of data.

    We use a bucketing technique from ``torchtext``. First, partition data in buckets of size
    100 * ``batch_size``. The examples inside the buckets are sorted using ``sort_key`` and batched.

    **Background**

        BucketBatchSampler is similar to a BucketIterator found in popular libraries like `AllenNLP`
        and `torchtext`. A BucketIterator pools together examples with a similar size length to
        reduce the padding required for each batch. BucketIterator also includes the ability to add
        noise to the pooling.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        data (iterable): Data to sample from.
        batch_size (int): Size of mini-batch.
        sort_key (callable): specifies a function of one argument that is used to extract a
          comparison key from each list element
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        biggest_batch_first (callable or None, optional): If a callable is provided, the sampler
            approximates the memory footprint of tensors in each batch, returning the 5 biggest
            batches first. Callable must return a number, given an example.

            This is largely for testing, to see how large of a batch you can safely use with your
            GPU. This will let you try out the biggest batch that you have in the data `first`, so
            that if you're going to run out of memory, you know it early, instead of waiting
            through the whole epoch to find out at the end that you're going to crash.

            Credits:
            https://github.com/allenai/allennlp/blob/3d100d31cc8d87efcf95c0b8d162bfce55c64926/allennlp/data/iterators/bucket_iterator.py#L43
        bucket_size_multiplier (int): Batch size multiplier to determine the bucket size.
        shuffle (bool, optional): If ``True``, the sampler will shuffle the batches.

    Example:
        >>> list(BucketBatchSampler(range(10), batch_size=3, drop_last=False))
        [[9], [3, 4, 5], [6, 7, 8], [0, 1, 2]]
        >>> list(BucketBatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    """

    def __init__(
            self,
            data,
            batch_size,
            drop_last,
            sort_key=_identity,
            bucket_size_multiplier=100,
            shuffle=True,
    ):
        self.sort_key = sort_key
        self.bucket_size_multiplier = bucket_size_multiplier
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data = data
        self.shuffle = shuffle

        self.bucket_size_multiplier = bucket_size_multiplier
        self.bucket_sampler = BatchSampler(
            RandomSampler(data), batch_size * bucket_size_multiplier, False)

    def __iter__(self):

        def get_batches():
            """ Get bucketed batches """
            for bucket in self.bucket_sampler: # bucket has (batch_size * bucket_size_multiplier) of indices
                for batch in ShuffleBatchSampler(
                        SortedSampler(bucket, lambda i: self.sort_key(self.data[i])),
                        self.batch_size,
                        drop_last=self.drop_last,
                        shuffle=self.shuffle):
                    batch = [bucket[i] for i in batch]

                    # Should only be triggered once
                    if len(batch) < self.batch_size and self.drop_last:
                        continue

                    yield batch

        return get_batches()

    def __len__(self):
        if self.drop_last:
            return len(self.data) // self.batch_size
        else:
            return math.ceil(len(self.data) / self.batch_size)


class BucketedSampler(Sampler):
    """ Sampler for bucketing """
    def __init__(self, data_source, bucketing_tgt=1):
        self.data_source = data_source
        if bucketing_tgt not in [1,2]:
            raise ValueError("bucketing target should be 1 or 2 (sentence1 or sentence2"
                             ", but got bucketing target={}".format(bucketing_tgt))
        self.bucketing_tgt = bucketing_tgt

    def __iter__(self):
        if self.bucketing_tgt == 1:
            sentence_length = self.data_source.question1_idx.map(len) # Series
        else:
            sentence_length = self.data_source.question2_idx.map(len)  # Series
        sorted_idx = sentence_length.sort_values(ascending=True).index.tolist()
        return iter(sorted_idx)

    def __len__(self):
        return len(self.data_source)


def collate_fn(inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    question1, question2, label = list(zip(*inputs))
    question1 = pad_sequence(question1, batch_first=True, padding_value=1)
    question2 = pad_sequence(question2, batch_first=True, padding_value=1)

    return question1, question2, torch.tensor(label)

