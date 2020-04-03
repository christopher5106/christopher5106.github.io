---
layout: post
title:  "Get FastText representation from pretrained embeddings with subword information"
date:   2020-04-02 05:00:00
categories: deep learning
---

FastText is a state of the art when speaking about non-contextual word embeddings. For that results account many optimizations, such as subword information, phrases, for which no documentation is available on how to reuse pretrained embeddings in our projects.

Let's see how to get a representation in Python.

First let's install FastText:

```
pip install fasttext
```

And download the [pretrained embeddings](https://fasttext.cc/docs/en/english-vectors.html), all of dimension 300:

```
mkdir /sharedfiles/fasttext

for FILE in wiki-news-300d-1M.vec.zip wiki-news-300d-1M-subword.vec.zip crawl-300d-2M.vec.zip crawl-300d-2M-subword.zip;
do
echo "Downloading $FILE";
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/$FILE -O /sharedfiles/fasttext/$FILE
unzip /sharedfiles/fasttext/$FILE -d /sharedfiles/fasttext/
rm /sharedfiles/fasttext/$FILE
done

https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.bin.gz
gunzip cc.en.300.vec.gz
mv gunzip cc.en.300.vec /sharedfiles/fasttext/
mv gunzip cc.en.300.bin /sharedfiles/fasttext/
```

import fasttext
ft = fasttext.load_model('/sharedfiles/fasttext/cc.en.300.bin')
model = fasttext.train_supervised('cooking.train', pretrainedVectors="/sharedfiles/fasttext/cc.en.300.vec", dim=300, epoch=0, maxn=5, minn=5)


In Python, let's import the libraries and use the function they offer us to load vectors:

```
import io
import fasttext
dim=300

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
```

Now on, we are left... to ourselves. It is hard to find any example on how to tokenize into words, compute subword information, and find phrases in Python. And even to be sure of the hyperparameters used in their embedding pretraining... such as `maxn`, `minn` or `wordNgrams` which are not specified in the model name or the documentation. The `gensim` package does not show neither how to get the subword information.

# Reverse engineering

There is no much possible available to reverse engineer. The idea is to get a groundtruth representation that we could reproduce by ourselves. One solution is to instantiate a supervised training of **0 epoch** for which it is possible to load the embeddings in the `.vec` format (word2vec format). For this, we need to provide it with a txt file 'cooking.train' to be read:

```
model = fasttext.train_supervised('cooking.train', pretrainedVectors="/sharedfiles/fasttext/wiki-news-300d-1M-subword.vec", dim=300, epoch=0)

>>> model.get_subwords("airplane")
(['airplane'], array([670412]))

>>> model.get_word_vector("airplane")
array([ 5.00e-04, -3.00e-04,  8.20e-03, -3.30e-03,  5.80e-03,  2.50e-03,
        6.90e-03, -3.40e-02,  1.70e-02,  2.24e-02,  1.31e-02, -1.49e-02,
       -1.00e-02,  8.20e-03, -8.80e-03,  9.50e-03,  2.02e-02, -1.13e-02,
        1.31e-02,  2.06e-02,  1.90e-02,  0.00e+00,  5.40e-03,  1.35e-02,
       -7.00e-03, -1.50e-02,  5.80e-03,  9.40e-03,  6.30e-03, -1.20e-03,
        1.14e-02,  4.60e-03,  2.95e-02, -5.80e-03,  5.80e-03,  2.40e-03,
       -1.28e-02, -2.20e-03,  2.56e-02, -2.00e-02, -1.57e-02, -5.08e-02,...
```

Note that the `get_subwords` method requires the model to be loaded. Normally we should have loaded `wiki-news-300d-1M.vec` under that setting, instead of `wiki-news-300d-1M-subword.vec`. But since we'd like to get a comparison of representation with the setting with subword information activated:

```
>>> model = fasttext.train_supervised('cooking.train', pretrainedVectors="/sharedfiles/fasttext/wiki-news-300d-1M-subword.vec", dim=300, wordNgrams=6, maxn=6, minn=3, epoch=0)

>>> model.get_subwords("airplane")
(['airplane', '<ai', '<air', '<airp', '<airpl', 'air', 'airp', 'airpl', 'airpla', 'irp', 'irpl', 'irpla', 'irplan', 'rpl', 'rpla', 'rplan', 'rplane', 'pla', 'plan', 'plane', 'plane>', 'lan', 'lane', 'lane>', 'ane', 'ane>', 'ne>'], array([ 670412, 1327487, 2536783, 2458297, 2015395, 1939857, 2022431,
      1463525, 2428216, 2532592, 1551336, 1258391, 2218115, 1833955,
      2876050, 1867068, 2346207, 1430146, 2591372, 2862031, 2658411,
      2170890, 1849641, 1854653, 2178739, 2828031, 2361248]))

>>> model.get_word_vector("airplane")
array([ 1.8584094e-04, -2.9165446e-04,  4.9583125e-04, -2.9015218e-04,
        5.4428069e-04, -4.5034866e-04,  4.8662224e-04, -1.3173044e-03,
        8.8238070e-04,  6.3112163e-04,  6.8609050e-04, -2.4462832e-04,
       -4.6237855e-04,  7.2012852e-05,  1.0311069e-04, -2.5771071e-05,
        1.6948332e-03, -4.2369164e-04,  6.3952326e-04,  6.2942918e-04,
        5.1321101e-04, -2.6033050e-04,  6.6818012e-04,  3.6886203e-04,
       -7.3394412e-04, -9.1407623e-04,  6.9407618e-04,  3.0970800e-04,...
```

As you can, the representation of the word `airplane` has changed: the subword information has been added.

# Subword information

Let's reproduce this in Python, removing dependency from model. First, let's modify the provided function to get indices as well:

```python
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = []
    indices = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        indices[tokens[0]] = len(data)
        data.append(list(map(float, tokens[1:])))
    return indices, np.stack(data)

indices, pretrained = load_vectors("/sharedfiles/fasttext/wiki-news-300d-1M-subword.vec")

minn=3
maxn=6

def get_subwords(word, indices, minn, maxn):
  _word = "<" + word + ">"
  _subwords = []
  if word in indices:
    _subwords.append(word)
  for ngram_start in range(0, len(_word)):
    for ngram_length in range(minn, maxn+1):
      if ngram_start+ngram_length <= len(_word):
        _candidate_subword = _word[ngram_start:ngram_start+ngram_length]
        if _candidate_subword not in _subwords:
          _subwords.append(_candidate_subword.replace(">", "").replace("<", ""))
  return _subwords

subwords = get_subwords("airplane", indices, minn, maxn)
print(subwords)

def get_representation(subwords, indices, pretrained):
  vector = 0
  for subword in subwords:
    if subword in indices:
      vector += pretrained[indices[subword]]
  return vector

print(get_representation(subwords, indices, pretrained))
```

# Tokenization into words


assumes to be given a single line of text. We split words on
                           # whitespace (space, newline, tab, vertical tab) and the control
                           # characters carriage return, formfeed and the null character.

# Phrases

```
>>> for w in model.get_words():
...     if w.count("-") == 5:
...             print(w)
...
b-----
--Kuzaar-T-C-
n-----
three-and-a-half-year-old
--The-G-Unit-Boss
He-Who-Must-Not-Be-Named
off-the-top-of-my-head
s-l-o-w-l-y
--None-of-the-Above
-----
f-----g
two-and-a-half-year-old
f-----
```

```
wordNgrams=4
```





--Sir
--My







"--------------------------------------------" in model.get_words()




-----------
-------------
---------
--------------------------------------------------sleeplessness
----------
buy-one-get-one-free
use-it-or-lose-it
------------------------------------
bull----
0-0-0-Destruct-0
-------------------------------------
three-and-a-half-hour
f-----g
--Animalparty--
English-as-a-second-language
-------------------------------
--------------------------------
four-and-a-half-year
two-and-a-half-year-old
Still-24-45-42-125
b----
Saint-Jean-Pied-de-Port
tell-it-like-it-is
B----
.----
----I
-----------------------------------
----------------------------------
--Jack-A-Roe
----The
---------User
--WFC--
----moreno
three-and-a-half-year





**Well done!**
