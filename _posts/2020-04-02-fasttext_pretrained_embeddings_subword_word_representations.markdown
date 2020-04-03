---
layout: post
title:  "Get FastText representation from pretrained embeddings with subword information"
date:   2020-04-02 05:00:00
categories: deep learning
---

FastText is a state-of-the art when speaking about **non-contextual word embeddings**. For that result, account many optimizations, such as subword information and phrases, but for which no documentation is available on how to reuse pretrained embeddings in our projects.

Let's see how to get a representation in Python.

First let's install FastText:

```
pip install fasttext
```

# Pretrained embeddings


Let's download the [pretrained embeddings](https://fasttext.cc/docs/en/english-vectors.html), all of dimension 300:

```
mkdir /sharedfiles/fasttext
for LANG in en fr;
do
echo "Downloading for lang $LANG";
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${LANG}.300.bin.gz
gunzip cc.${LANG}.300.bin.gz
mv cc.${LANG}.300.bin /sharedfiles/fasttext/
done
```

And load a model:

```python
>>> import fasttext

>>> ft = fasttext.load_model('/sharedfiles/fasttext/cc.en.300.bin')

>>> ft.get_words()[:10]

[',', 'the', '.', 'and', 'to', 'of', 'a', '</s>', 'in', 'is']

>>> len(ft.get_words())
2000000

>>> input_ = ft.get_input_matrix()

>>> input_.shape
(4000000, 300)
```

The input matrix contains an embedding reprentation for 4 million word and subwords. The number of words in the vocabulary is 2 million.

First thing you might notice, subword embeddings are not available in  the released `.vec` text dumps in `word2vec` format:

```
$ wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
$ gunzip cc.en.300.vec.gz
$ wc -l cc.en.300.vec
2000001
$ head -1 /sharedfiles/fasttext/cc.en.300.vec
2000000 300
```

The first line in the file specifies 2 m words and 300 dimension embeddings, and the remaining 2 million lines is a dump of the embeddings.

In this document, I'll explain how to dump the full embeddings and use them in a project.


# Dump the full embeddings from .bin file

As seen in previous section, you need to load the model first from the `.bin` file:

```python
import os
import fasttext
import numpy as np


def save_embeddings(model, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  np.save(os.path.join(output_dir, "embeddings"), model.get_input_matrix())
  with open(os.path.join(output_dir, "vocabulary.txt"), "w", encoding='utf-8') as f:
    for word in model.get_words():
      f.write(word+"\n")


for lang in ["en", "fr"]:
  ft = fasttext.load_model('/sharedfiles/fasttext/cc.' + lang + '.300.bin')
  save_embeddings(ft, '/sharedfiles/fasttext/cc.' + lang + '.300')
```


# Getting a word representation


Now, you should be able to load full embeddings and get a word representation directly in Python:

```python
def load_embeddings(output_dir):
  input_matrix = np.load(os.path.join(output_dir, "embeddings.npy"))
  words = []
  with open(os.path.join(output_dir, "vocabulary.txt"), "r", encoding='utf-8') as f:
    for line in f.readlines():
      words.append(line)
  return words, input_matrix

vocabulary, embeddings = load_embeddings('/sharedfiles/fasttext/cc.en.300')

```



In Python, let's import the libraries and use the function they offer us to load vectors:

```
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
