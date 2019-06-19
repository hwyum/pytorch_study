# NLP Implementation in PyTorch

## PyTorch basic
**1. wk1: pytorch official toturial**
  * [What is torch.nn really?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
  * Data Loading and processing tutorial
  * Saving and loading models


**2. wk2: CIFAR10 CNN Implementation with custom DataLoader**

## Classification
**[3. wk3: Naver Movie Classification](https://github.com/hwyum/pytorch_study/tree/master/wk3_movie_classification)**
 * Reference: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

**[4. wk4: Naver Movie Classification in Character level](https://github.com/hwyum/pytorch_study/tree/master/wk4_CharacterCNN)**
 * Reference: [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)

**[5. wk5: Naver movie Classification with Very Deep CNN](https://github.com/hwyum/pytorch_study/tree/master/wk5_VeryDeepCNN)**
* Reference: [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/abs/1606.01781) 

**[6. wk6: Naver movie Classification with ConvRec](https://github.com/hwyum/pytorch_study/tree/master/wk6_ConvRec)**
* Reference: [Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](https://arxiv.org/abs/1602.00367)

## Q-Q Similarity
**[7. wk7: Q-Q Similarity with SAN](https://github.com/hwyum/pytorch_study/tree/master/wk7_SAN)**
* Dataset: [Korean Question Pair Dataset (by songys)](https://github.com/songys/Question_pair)
* Reference: [A Structured Self-attentive Sentence Embedding (2017)](https://arxiv.org/abs/1703.03130)

## NER
**[8. wk8: NER with BiLSTM-CRF](https://github.com/hwyum/pytorch_study/tree/master/wk8_BiLSTM-CRF)**
* Dataset: [Naver NLP Challenge Dataset](https://github.com/naver/nlp-challenge/tree/master/missions/ner)
* Reference: [Bidirectional LSTM-CRF Models for Sequence Tagging (2015)](https://arxiv.org/abs/1508.01991) <br>
* Result: 15 Epochs

    |training loss|validation score|validation f1 score|
    |:-----------:|:--------------:|:-----------------:|
    |1.2          |127.940         |0.956              |

## Neural Machine Translation (NMT)
**9. wk9: NER with BiLSTM-CRF**
* Dataset: [Korean-English Parallel Corpus](https://sites.google.com/site/koreanparalleldata/)
* Reference: [Effective Approaches to Attention-based Neural Machine Translation
](https://arxiv.org/abs/1508.04025) <br>
