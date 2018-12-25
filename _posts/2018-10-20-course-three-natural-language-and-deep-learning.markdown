---
layout: post
title:  "Course 3: natural language and deep learning!"
date:   2018-10-20 10:00:00
categories: deep learning
---

Here is my course of deep learning in 5 days only! You might first check [Course 0: deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html), [Course 1: program deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-one-programming-deep-learning.html) and [course 2: build deep learning networks!](http://christopher5106.github.io/deep/learning/2018/10/20/course-two-build-deep-learning-networks.html) if you have not read them.

In this article, I develop techniques for natural language. Tasks include text classification (finding the sentiment, positive or negative, the language, ...), segmentation (POS taging, Named Entities extraction ,...), translation, ... as in computer vision.

But, while computer vision deals with **continuous** (pixel values) and **fixed-length** inputs (images, usually resized or cropped to a fixed dimension), natural language consists of **variable-length** sequences (words, sentences, paragraphs, documents) of **discrete** inputs, either characters or words, belonging to a fixed size dictionary (the alphabet or the vocabulary respectively), depending if we work at character level or word level.

There are two challenges to overcome :

- transforming discrete inputs into continuous representations or vectors

- transforming variable-length sequences into a fixed-length representations


# The dictionary

Texts are sequences of characters or words, depending if we word at character level or word level. It is possible to work at both levels and concatenate the representations at higher level in the neural network. The **dictionary** is a fixed-size list of symbols found in the input data: at character level, we call it an alphabet, while at word level it is usually called a vocabulary. While the words incorporate a semantic meaning, the vocabulary has to be limited to a few tens thousands entries, so many out-of-vocabulary tokens cannot be represented.  

There exists a better encoding for natural language, the **Byte-Pair-Encoding (BPE)**, a compression algorithm that iteratively replaces the most frequent pairs of symbols in sequences by a new symbol: initially, the symbol dictionary is composed of all characters plus the 'space' or 'end-of-word' symbol, then recursively count each pair of symbols (without crossing word boundaries) and replace the most frequent pair by a new symbol.

For example, if ('T', 'H') is the most frequent pair of symbols, we replace all instances of the pair by the new symbol 'TH'. Later on, if ('TH', 'E') is most frequent pair, we create a new symbol 'THE'. The process stops when the desired target size for the dictionary has been achieved. At the end, the symbols composing the dictionary are essentially characters, bigrams, trigrams, as well as most common words or word parts.  

The advantage of BPE is to be capable of **open-vocabulary** through a fixed-size vocabulary though, while still working at a coarser level than characters. In particular, names, compounds, loanwords which do not belong to a language word vocabulary can still be represented by BPE. A second advantage is that the BPE preprocessing works for all languages.

In translation, better results are achieved by joint BPE, encoding both the target and source languages with the same dictionary of encoding. For languages using a different alphabets, characters are transliterated from one alphabet to the other. This helps in particular to copy Named Entities which do not belong to a dictionary.

Last, it is possible to relax the greedy and deterministic symbol replacement of BPE, by using a **unigram language model** that assumes that each symbol is an unobserved latent variable of the sequence and occurs independently in the sequence. Given this assumption, the probability of a sequence of symbols $$ (x_1, ..., x_M) $$ is given by :

$$ P(x_1, ..., x_M) = \prod_{i=1}^M p(x_i) $$

and the probability of a sentence or text S to occur is given by the sum of probabilities of each encodings:

$$ P(S) = \sum_{(x_1, ..., x_M)==S} P(x_1, ..., x_M) $$

So it is possible to compute a dictionary of the desired size that maximizes (locally) the likelihood by an iterative algorithm starting from a huge dictionary of most frequent substring, estimating the expectation as in the EM algorithm, and removing the subwords with less impact on the likelihood. Also, multiple decodings into a sequence of symbols are possible for a text and the model gives each of them a probability. At training time, it is possible to sample a decoding of the input given the symbol distribution. At inference, it is possible to compute the predictions using multiple decodings, and choose the most confident prediction. Such a technique, called **subword regularization**, augments the training data stochastically and improves accuracy and robustness in natural language tasks.

Have a look at [SentencePiece](https://github.com/google/sentencepiece). Note that 'space' character is treated as a symbol and pretokenization of the sequences is not necessary.  

# Distributed representations for the symbols of the dictionary




# Distributed representations of the sentences, paragraphs or documents


# Metrics for translation

Models are trained to maximize the log likelihood, but this does not give any idea of the final quality of the model. Since human evaluations is costly and long, metrics have been developped that correlates well with human judgments, on a given set of possible translations (reference translations) for the task of translation for example.

The ChrF3 is one of them, the most simple one, is the [F3-score](http://christopher5106.github.io/deep/learning/2018/10/20/course-one-programming-deep-learning.html#training-curves-and-metrics) for the character n-grams. The best correlations are obtained with $$ n=6 $$. The ChrF3 has a recall bias.

Another metric, BLUE, has a precision bias. It is the precision for the character n-grams, where the count for each n-gram matches is clipped, not to be bigger than the number of n-grams in the reference translation, in order to penalize systems that generate a same word multiple times. The metric can be case-sensitive, in order to take into account the Named Entities for example.

ROUGE-n is the recall equivalent metric.

There exists metrics on words (called unigrams) also, such as

- WER (Word Error Rate), the Levenstein distance at word level.

- METEOR, the ChrF3 equivalent at word level, reweighted by a non-alignment penalty.


# Under construction

a recurrent network is a feedforward network with two inputs
hidden information
<img src="{{ site.url }}/img/deeplearningcourse/DL21.png">


Normalize attention by $$ \frac{1}{\sqrt{d}} $$

unsupervised nmt
