---
layout: post
title:  "Course 3: natural language and deep learning!"
date:   2018-10-20 10:00:00
categories: deep learning
---

Here is my course of deep learning in 5 days only! You might first check [Course 0: deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-zero-deep-learning.html), [Course 1: program deep learning!](http://christopher5106.github.io/deep/learning/2018/10/20/course-one-programming-deep-learning.html) and [course 2: build deep learning networks!](http://christopher5106.github.io/deep/learning/2018/10/20/course-two-build-deep-learning-networks.html) if you have not read them.

In this article, I develop techniques for natural language.

While computer vision deals with **continuous** (pixel values) and **fixed-length** inputs (images, usually resized or cropped to a fixed dimension), natural language consists of **variable-length** sequences (words, sentences, paragraphs, documents) of **discrete** inputs, either characters or words, belonging to a fixed size dictionary (the alphabet or the vocabulary respectively), depending if we work at character level or word level.

There are two challenges to overcome :

- transforming discrete inputs into continuous representations or vectors

- transforming variable-length sequences into a fixed-length representations


# Discrete input representations

Texts are sequences of characters or words, depending if we word at character level or word level. It is possible to work at both levels and concatenate the representations. There exists a third case of encoding for natural language, the **Byte-Pair-Encoding (BPE)**, a compression algorithm that iteratively replaces the most frequent pairs of symbols in sequences by a new symbol: initially, the symbol vocabulary is composed of all characters plus the 'space' or 'end-of-word' symbol, then recursively count each pair of symbols (without crossing word boundaries) and replace the most frequent pair by a new symbol.

For example, if ('T', 'H') is the most frequent pair of symbols, we replace all instances of the pair by the new symbol 'TH'. Later on, if ('TH', 'E') is most frequent pair, we create a new symbol 'THE'. The process stops when the desired target size for the vocabulary has been achieved. At the end, the symbols composing the vocabulary are essentially characters, bigrams, trigrams, as well as most common words or word parts.  

The advantage of BPE is to be capable of open-vocabulary through a fixed-size vocabulary, while still working at a coarser level than characters. In particular, names, compounds, loanwords which do not belong to a language word vocabulary can still be represented by BPE.

In translation, better results are achieved by joint BPE, encoding both the target and source languages with the same dictionary of encoding. For languages using a different alphabets, characters are transliterated from one alphabet to the other. This helps in particular to copy Named Entities which do not belong to a dictionary.



# Metrics

ChrF3 has recall

BLUE score has a precision bias


# Under construction

a recurrent network is a feedforward network with two inputs
hidden information
<img src="{{ site.url }}/img/deeplearningcourse/DL21.png">


Normalize attention by $$ \frac{1}{\sqrt{d}} $$
