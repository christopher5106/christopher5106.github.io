---
layout: post
title:  "About Bayes"
date:   2016-09-17 17:00:51
categories: machine learning
---

Let's speak about Bayes : its theorem, Bayes classifiers and Bayes inference.

# Bayes theorem

The conditional probabilities $$ P( B | A  ) $$ and $$ P( A | B  ) $$ verify :

$$ P( A \cap B ) = P( A | B ) P(B) =  P( B | A ) P(A) $$

meaning that the probability to see A and B, is the probability to see B, and then the probability to A given we have already seen B. Or inversely, seeing A, then B given A.

This leads to the Bayes theorem that gives the relation between $$ P( B | A  ) $$ and $$ P( A | B  ) $$ :

$$ P( A | B ) = \frac{ P( B | A  ) P( A )  }{ P( B )} $$

Let's see what it means in practice.


# Bayes updating and inference

Suppose we want to predict the class y of an observation x.

$$ f(y) = p(y | x ) = \frac{ P( x | y  ) P( y )  }{ P( x )} $$

which is rewritten :

$$ \txt{posterior} = \frac{ \txt{likelihood} \times \txt{prior}  }{ \txt{evidence }} $$

The prior is the probability of the class when we do not observe x. It is the distribution of classes.

Bayes theorem tells us that once we made the observation x, the probability of the class (the posterior) has evolved and gives us an update rule.


# Naive bayes classifier

Suppose the x is represented by n independant feature conditionaly to the class:

$$ x = (x_1, ... , x_n) $$

$$ P(x | y ) = \prod_i p(x_i|y) $$

then

$$ f(y) ~ p(y)  \prod_i p(x_i | y) $$

because $$ p(x) $$ is a constant if features are known.

The naïve Bayes classifier consists in using the maximum a posteriory rule :

$$ ŷ = \argmax_c p(y_c)  \prod_i p(x_i | y_c) $$

The class prior might be easy to estimate from the training set :

$$ p(y_c) = \frac{ \txt{number of examples of the class c} }{ \txt{size of the set} } $$

To estimate the distribution of features for each class, assumptions are made with models.
