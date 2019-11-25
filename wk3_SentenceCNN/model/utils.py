import numpy as np
from collections import Counter
from typing import Optional, List, Dict, Hashable, Callable, Union

UNK_IDX = 0

class Vocab:
    """
    Reference: gluonnlp.vocab (https://gluon-nlp.mxnet.io/_modules/gluonnlp/vocab/vocab.html#Vocab)
    Indexing and embedding attachment for text tokens.
    Args:
        counter: Counts text token frequencies in the text data. Its keys will be indexed according to
                 frequency threshold such as `max_size` and `min_freq`.

    Attributes:
        embedding
        idx_to_token(list)
        reserved_tokens(list of strs or None): A list of reserved tokens that will always be indexed
        token_to_idx(dict mapping str to int): A dict mapping each token to its index integer
        max_size:
            The maximum possible number of the most frequent tokens in the keys of `counter` that can be
            indexed. Note that this argument does not count any token from `reserved_tokens`.

    """
    def __init__(self, counter: Optional[Counter] = None, max_size: Optional[int] = None,
                 min_freq: int = 1, unknown_token: Optional[Hashable] = '<unk>',
                 padding_token: Optional[Hashable] = '<pad>',
                 bos_token: Optional[Hashable] = '<bos>',
                 eos_token: Optional[Hashable] = '<eos>',
                 reserved_tokens: Optional[List[Hashable]] = None,
                 token_to_idx: Optional[Dict[Hashable, int]] = None):
        """ initialization of vocab class
        Args:
            counter:
                Counts text token frequencies in the text data. Its keys will be indexed according to
                frequency thresholds such as `max_size` and `min_freq`. Keys of `counter`,
                `unknown_token`, and values of `reserved_tokens` must be of the same hashable type.
                Examples: str, int, and tuple.
            max_size:
                The maximum possible number of the most frequent tokens in the keys of `counter` that
                can be indexed
            min_freq:
                The minumum frequency required for a token in the keys of `counter` to be indexed.

            """

        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'

        # Set up idx_to_token and token_to_idx based on presence of unknown token
        self._unknown_token = unknown_token
        self._padding_token = padding_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._reserved_tokens = reserved_tokens
        self._idx_to_token = [unknown_token] if unknown_token else []
        self._token_to_idx = {}

        for token in [
            self._unknown_token,
            self._padding_token,
            self._bos_token,
            self._eos_token
        ]:
            if token:
                self._idx_to_token.append(token)
                self._token_to_idx.update((token, idx) for idx, token in enumerate(self._idx_to_token))

        if reserved_tokens:
            assert len(set(reserved_tokens)) == len(reserved_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens'
            self._idx_to_token.extend(reserved_tokens)
            self._token_to_idx.update((token, idx) for idx, token in enumerate(self._idx_to_token))

        self._embedding = None

        if counter:
            self._index_counter_keys(counter, unknown_token, reserved_tokens, max_size, min_freq)

        if token_to_idx:
            self._sort_index_according_to_user_specification(token_to_idx)

    def _index_counter_keys(self, counter, unknown_token, reserved_tokens, max_size, min_freq):
        """ Indexes keys of `counter` """
        unknown_and_reserved_tokens = set(reserved_tokens) if reserved_tokens else set()

        if unknown_token:
            unknown_and_reserved_tokens.add(unknown_token)

        token_freqs = sorted(counter.items(), key=lambda x:x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(unknown_and_reserved_tokens) + (len(counter) if not max_size else max_size)

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            if token not in unknown_and_reserved_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self._token_to_idx.keys()):
            raise ValueError('User-specified token_to_idx mapping can only contain '
                             'tokens that will be part of the vocabulary.')
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError('User-specified indices must not contain duplicates.')
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(self._token_to_idx):
            raise ValueError('User-specified indicies must not be <0 or >= the number of tokens '
                             'that will be in the vocabulary. The current vocab contains {}'
                             'tokens'.format(len(self._token_to_idx)))

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self._token_to_idx[token]
            token_tobe_replaced = self._idx_to_token[new_idx]

            self._token_to_idx[token] = new_idx
            self._token_to_idx[token_tobe_replaced] = old_idx
            self._idx_to_token[old_idx] = token_tobe_replaced
            self._idx_to_token[new_idx] = token

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, array):
        self._embedding = array

    def __contains__(self, token):
        """ Checks whether a text token exists in the vocabulary. """
        return token in self._token_to_idx

    def __getitem__(self, tokens):
        """ Looks up indices of text tokens according to the vocabulary. """
        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    # Todo: To be implemented
    def set_embedding(self, *embeddings):
        # """ Attaches one or more embeddings to the indexed text tokens """
        # if len(embeddings) == 1 and embeddings[0] is None:
        #     self._embedding = None
        #     return
        #
        # for embs in embeddings:
        #     assert isinstance(embs, emb.)
        pass

    def to_indices(self, tokens):
        """ Looks up indices of text tokens
        Args:
            tokens(Union[str, List[str]): a source token or tokens to be converted
        Returns:
            Uinon(int, List[int]): converted indices of source token or tokens
        """
        if isinstance(tokens, list):
            return [
                self._token_to_idx[token]
                if token in self._token_to_idx
                else self._token_to_idx[self._unknown_token]
                for token in tokens
            ]
        else:
            return (
                self._token_to_idx[tokens]
                if tokens in self._token_to_idx
                else self._token_to_idx[self._unknown_token]
            )


    def to_tokens(self, indices):
        """ Converts token indices to tokens according to the vocabulary
        Args:
            indices(Union[int, List[int]): A source token index or token indices to be converted
        Returns:
            Union(str, List[str]): A token or a list of tokens according to the vocabulary.
        """
        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indicies = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError('Token index {} in the provided `indicies` is invalid'.format(idx))
            tokens.append(self._idx_to_token[idx])

        return tokens[0] if to_reduce else tokens


class Tokenizer:
    """ Class for tokenizing and transfoming to indices given vocab and split function """
    def __init__(self, vocab: Vocab, split_fn:Callable):
        self.vocab = vocab
        self._split_fn = split_fn

    def __call__(self, sent:str):
        """ return tokens generated by split function """
        return self._split_fn(sent)

    def transform(self, tokens:Union[List, str]):
        """ return indicies of tokens generated by split function in advance """
        if not isinstance(tokens, Union[list, str]):
            raise TypeError("tokens should be a single str or a list")

        indicies = self.vocab.to_indices(tokens)
        return indicies

    def tokenize_and_transform(self, sent:str):
        """ return indicies given sentence (first transform to tokens and to indicies """
        tokens = self._split_fn(sent)
        indicies = self.vocab.to_indicies(tokens)
        return indicies


class PadSequence:
    """ Pad the sequence.
    Pad the sequence to the given `length` by inserting `pad_val`. If `clip` is set,
    sequence that has length larger than `length` will be clipped
    Args:
        length(int): The maximum length to pad/clip the sequence
        pad_val(int): The pad value. Default: 0
        clip(bool): whether or not clip the sentence if sentence has length larger than given length. Default: True
    """
    def __init__(self, length:int, pad_val:int, clip:bool=True):
        self._length = length
        self._pad_val = pad_val
        self._clip = clip

    def __call__(self, sample):
        """
        Args:
            sample: list of number or numpy array (np.ndarray)
        Returns:
             list of number or numpy array (np.ndarray)
        """
        sample_length = len(sample)
        if sample_length >= self._length:
            if self._clip and sample_length > self._length:
                return sample[:self._length]
            else:
                return sample

        else:
            if isinstance(sample, np.ndarray):
                pad_width = [(0, self._length - sample_length)] + \
                            [(0, 0) for _ in range(sample.ndim - 1)]
                return np.pad(sample,
                              mode='constant',
                              constant_values=self._pad_val,
                              pad_width=pad_width)
            elif isinstance(sample, list):
                return sample + [
                    self._pad_val for _ in range(self._length - sample_length)
                ]
            else:
                raise NotImplementedError(
                    'The input mst be 1) list or 2) numpy.ndarray, received type={}'.format(str(type(sample)))
                )

