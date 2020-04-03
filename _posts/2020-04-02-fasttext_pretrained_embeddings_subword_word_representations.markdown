---
layout: post
title:  "Get FastText representation from pretrained embeddings with subword information"
date:   2020-04-02 05:00:00
categories: deep learning
---

FastText is a state-of-the art when speaking about **non-contextual word embeddings**. For that result, account many optimizations, such as subword information and phrases, but for which no documentation is available on how to reuse pretrained embeddings in our projects. The `gensim` package does not show neither how to get the subword information.

Let's see how to get a representation in Python.

First let's install FastText:

```
pip install fasttext
```

# Pretrained embeddings


Let's download the [pretrained unsupervised models](https://fasttext.cc/docs/en/english-vectors.html), all producing a representation of dimension 300:

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

And load one of them for example, the english one:

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



# Loading the embeddings


Now, you should be able to load full embeddings and get a word representation directly in Python:

```python
def load_embeddings(output_dir):
  input_matrix = np.load(os.path.join(output_dir, "embeddings.npy"))
  words = []
  with open(os.path.join(output_dir, "vocabulary.txt"), "r", encoding='utf-8') as f:
    for line in f.readlines():
      words.append(line.rstrip())
  return words, input_matrix

vocabulary, embeddings = load_embeddings('/sharedfiles/fasttext/cc.en.300')
```

# Getting a word representation with subword information


The first function required is a hashing function to get row indice in the matrix for a given subword:

```python
def get_hash(subword, bucket=2000000, nb_words=2000000):
  h = 2166136261
  for c in subword:
    c = ord(c) % 2**8
    h = (h ^ c) % 2**32
    h = (h * 16777619) % 2**32
  return h % bucket + nb_words
```

In the model loaded, subwords have been computed from 5-grams of words. My implementation might differ a bit from [original](https://github.com/facebookresearch/fastText/blob/7842495a4d64c7a3bb4339d45d6e64321d002ed8/src/dictionary.cc#L172) for special characters:

```python
def get_subwords(word, vocabulary, minn=5, maxn=5):
  _word = "<" + word + ">"
  _subwords = []
  _subword_ids = []
  if word in vocabulary:
    _subwords.append(word)
    _subword_ids.append(vocabulary.index(word))
  for ngram_start in range(0, len(_word)):
    for ngram_length in range(minn, maxn+1):
      if ngram_start+ngram_length <= len(_word):
        _candidate_subword = _word[ngram_start:ngram_start+ngram_length]
        if _candidate_subword not in _subwords:
          _subwords.append(_candidate_subword)
          _subword_ids.append(get_hash(_candidate_subword))
  return _subwords, np.array(_subword_ids)
```

Now it is time to compute the vector representation, following the [code](https://github.com/facebookresearch/fastText/blob/02c61efaa6d60d6bb17e6341b790fa199dfb8c83/src/fasttext.cc#L103), the word representation is given by:

$$ \frac{1}{\| N \|   + 1 } * (v_w +  \sum_{n \in N} x_n ) $$

where N is the set of n-grams for the word, $$x_n$$ their embeddings, and $$v_n$$ the word embedding if the word belongs to the vocabulary.

```python
def get_word_vector(word, vocabulary, embeddings):
  subwords = get_subwords(word, vocabulary)
  return np.mean([embeddings[s] for s in subwords[1]], axis=0)
```

<span style="color:red">Q1: It looks different from the [paper, section 2.4](https://arxiv.org/pdf/1712.09405.pdf):
$$ v_w + \frac{1}{\| N \|} \sum_{n \in N} x_n $$
</span>

Let's test everything now:

```python
subwords = get_subwords("airplane", vocabulary)
print(subwords)
print(get_word_vector("airplane", vocabulary, embeddings).shape)
```

returns `(['airplane', '<airp', 'airpl', 'irpla', 'rplan', 'plane', 'lane>'], array([  11788, 3452223, 2457451, 2252317, 2860994, 3855957, 2848579]))` and an embedding representation for the word of dimension `(300,)`.

# Check

Let's check if reverse engineering has worked and compare our Python implementation with the Python-bindings of the C code:

```
>>> ft = fasttext.load_model('/sharedfiles/fasttext/cc.en.300.bin')

>>> ft.words == vocabulary
True

>>> np.allclose(ft.get_input_matrix(), embeddings)
True

>>> for word in ["airplane", "see", "qklsmjf", "qZmEmzqm"]:
  print("Subwords:", get_subwords(word, vocabulary)[0] == ft.get_subwords(word)[0])
  print("Subword_ids:", np.allclose(get_subwords(word, vocabulary)[1], ft.get_subwords(word)[1]))
  print("Representations:", np.allclose(get_word_vector(word, vocabulary, embeddings), ft.get_word_vector(word)))

Subwords: True
Subword_ids: True
Representations: True
Subwords: True
Subword_ids: True
Representations: True
Subwords: True
Subword_ids: True
Representations: True
Subwords: True
Subword_ids: True
Representations: True
```

Everything is correct.

# Tokenization into words


assumes to be given a single line of text. We split words on
whitespace (space, newline, tab, vertical tab) and the control
characters carriage return, formfeed and the null character.

```python
def tokenize(sentence):
  pass

```
TODO

# Phrases

Looking at the vocabulary, it looks like "-" is used for phrases (i.e. word N-grams) and it won't harm to consider so. Setting `wordNgrams=4` is largely sufficient, because above 5, the phrases in the vocabulary do not look very relevant:

```
>>> for w in ft.get_words():
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

<span style="color:red">Q2: what is the hyperparameter used for wordNgrams in the released models ?</span>

Let's extract word Ngrams from a text:

```python
def extract_wordNgrams(sentence):
  pass
```
TODO

<span style="color:red">Q3: How is the phrase embedding integrated in the final representation ? Is it a simple addition ?</span>

<span style="color:red">Q4: I'm wondering if the words "--Sir" and "--My" I find in the vocabulary have a special meaning. In the meantime, when looking at words with more than 6 characters "-", it looks very strange. I'm wondering if this could not have been removed from the vocabulary:
</span>

```
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
```

You can test it by asking: `"--------------------------------------------" in ft.get_words()`. The answer is `True`.

**Well done!**
