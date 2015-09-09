---
layout: post
title:  "Evaluate linear model, geometric brownian motion (Black-Scholes-Merton model), square-root diffusion, stochastic volatility, with Monte Carlo Simulations for Value at Risk and Credit Value Ajustements"
date:   2015-09-09 23:00:51
categories: finance
---

In this document, estimates of the value at risk and credit value adjustments are done with Monte Carlo simulations.

- **A VaR (value at risk) of 1 million with a 5% p-value et 2 weeks** is 5% of chance to lose 1 million over 2 weeks.

- **A CVaR (conditional value at risk) of 1 millions with 5% q-value and 2 weeks** is when the average expected loss in the worst 5% of outcomes is 1 million.

- A **Monte Carlo simulation** is a simple way to estimate some variables (such as the p-value over the returns) on an instrument, when the law of the instrument is known, by simulating the instrument over a period of time, doing this simulation N times which are **N trials**, or **N paths**, in order to compute the variables. Monte Carlo simulation are computation intensive.

Let's see some well-known laws in practice :) ...

#A linear model

It's a very simple model where the relation between market factors (such as indexes) and market returns of an instrument (such as stocks) is given by a linear model over the features of the market factors :

    market returns = linear( market features )

where

- market features = a transformation (such as derivative for market moves, or other functions …) over the market factors

- market factor returns follow a  multivariate normal distribution since market factors are often correlated.

An example can be found in *Spark Advanced Analytics* book :

![Spark Advanced Analytics]({{ site.url }}/img/advanced_analytics_spark.gif)

{% highlight bash %}
cd ~/examples
git clone https://github.com/sryza/aas
cd aas
mvn install
cd ch09-risk
#download all stocks in the NASDAQ index :
./download-all-symbols.sh

#download indexes SP500 et Nasdaq :
mkdir factors
./download-symbol.sh SNP factors
./download-symbol.sh NDX factors
{% endhighlight %}
