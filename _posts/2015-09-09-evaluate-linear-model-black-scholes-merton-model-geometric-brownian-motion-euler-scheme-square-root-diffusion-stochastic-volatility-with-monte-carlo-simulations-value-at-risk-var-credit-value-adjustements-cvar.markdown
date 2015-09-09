---
layout: post
title:  "Evaluate linear model, geometric brownian motion (Black-Scholes-Merton model), square-root diffusion, stochastic volatility, with Monte Carlo Simulations for Value at Risk and Credit Value Adjustements"
date:   2015-09-09 23:00:51
categories: finance
---

In this document, estimates of the value at risk and credit value adjustments are done with Monte Carlo simulations.

- **A VaR (value at risk) of 1 million with a 5% p-value et 2 weeks** is 5% of chance to lose 1 million over 2 weeks.

- **A CVaR (conditional value at risk) of 1 millions with 5% q-value and 2 weeks** is when the average expected loss in the worst 5% of outcomes is 1 million.

- A **Monte Carlo simulation** is a simple way to estimate some variables (such as the p-value over the returns) on an instrument, when the law of the instrument is known, by simulating the instrument over a period of time, doing this simulation N times which are **N trials**, or **N paths**, in order to compute the variables. Monte Carlo simulation are computationally intensive and require distributed computation (with *Spark* technology for example) or optimized computation (with *numpy, numexpr, cython  or numba* libraries in Python).

Let's see some well-known laws in practice :) ...

#A linear model on normally distributed market factor returns

It's a very simple model where the relation between market factors (such as indexes) and market returns of an instrument (such as stocks) is given by a linear model over the features of the market factors :

    market returns = linear( market features )

where

- market features = a transformation (such as derivative for market moves, or other functions …) over the market factors

- market factor returns follow a  multivariate normal distribution since market factors are often correlated.

I take this example from ["Advanced Analytics with Spark" by Sandy Ryza, Uri Laserson, Sean Owen and Josh Wills (O'Reilly)."](http://shop.oreilly.com/product/0636920035091.do). I won't explain the book, but give details on how to reproduce the results.

{% highlight bash %}
#download the code (commit eed30d7214e0e7996068083f2ac6793e6375a768)
cd ~/examples
git clone https://github.com/sryza/aas
cd aas
mvn install
cd ch09-risk/data

#download all stocks in the NASDAQ index :
./download-all-symbols.sh

#download indexes SP500 et Nasdaq :
mkdir factors
./download-symbol.sh SNP factors
cd factors
wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/data/factors/NDX.csv
wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/data/factors/crudeoil.tsv
wget https://s3-eu-west-1.amazonaws.com/christopherbourez/public/data/factors/us30yeartreasurybonds.tsv
cd ../..

#launch a cluster of 5 instances (1 master and 4 slaves)
~/technologies/spark-1.4.1-bin-hadoop2.6/ec2/spark-ec2 -k sparkclusterkey -i ~/sparkclusterkey.pem --region=eu-west-1 --copy-aws-credentials --instance-type=m1.large -s 4 --hadoop-major-version=2 launch spark-cluster

#submit the job
master=ec2-IP.REGION.compute.amazonaws.com
~/technologies/spark-1.4.1-bin-hadoop2.6/bin/spark-submit --executor-memory 6g --driver-memory 6g --driver-java-options "-Duser.country=UK -Duser.language=en" --conf "spark.executor.extraJavaOptions=-Duser.country=UK -Duser.language=en" --class com.cloudera.datascience.risk.RunRisk --master spark://${master}:7077 --deploy-mode client target/ch09-risk-1.0.0-jar-with-dependencies.jar
{% endhighlight %}

The code is explained in details in the ["Advanced Analytics with Spark" by Sandy Ryza, Uri Laserson, Sean Owen and Josh Wills (O'Reilly)."](http://shop.oreilly.com/product/0636920035091.do). The script computes the multivariate normal distribution parameters for the market factor returns, the linear model between market factor returns and market returns, N trials distributed on the cluster, compute the VAR/CVAR and evaluate the confidence interval and the Kupiec's proportion of failures (POF) for the computed VAR and CVAR.


Rather than submitting the whole class, you can also submit step-by-step the instructions in the spark-shell :

{% highlight bash %}
#Launch the shell
~/technologies/spark-1.4.1-bin-hadoop2.6/bin/spark-shell --executor-memory 6g --driver-memory 6g --driver-java-options "-Duser.country=UK -Duser.language=en" --conf "spark.executor.extraJavaOptions=-Duser.country=UK -Duser.language=en" --jars target/ch09-risk-1.0.0-jar-with-dependencies.jar --master spark://ec2-52-19-187-119.eu-west-1.compute.amazonaws.com:7077
{% endhighlight %}

In the shell :

{% highlight bash %}
import com.cloudera.datascience.risk._
import com.cloudera.datascience.risk.RunRisk._
import java.io.File
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression
import org.apache.commons.math3.stat.correlation.Covariance
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation
import com.github.nscala_time.time.Imports._
import breeze.plot._
val fiveYears = 260 * 5+10
val start = new DateTime(2009, 10, 23, 0, 0)
val end = new DateTime(2014, 10, 23, 0, 0)
val stocks1 = readHistories(new File("./data/stocks/")).filter(_.size >=
fiveYears)
val stocks = stocks1.map(trimToRegion(_, start, end)).map(fillInHistory(_, start
, end))
val factorsPrefix = "./data/factors/"
val factors1 = Array("crudeoil.tsv", "us30yeartreasurybonds.tsv").map(x => new File(factorsPrefix + x)).map(readInvestingDotComHistory)
val factors2 = Array("SNP.csv", "NDX.csv").map(x => new File(factorsPrefix + x)).map(readYahooHistory)
val factors = (factors1 ++ factors2).map(trimToRegion(_, start, end)).map(fillInHistory(_, start, end))
val stocksReturns = stocks.map(twoWeekReturns)
val factorsReturns = factors.map(twoWeekReturns)
val factorMat = factorMatrix(factorsReturns)
val models = stocksReturns.map(linearModel(_, factorMat))
val rSquareds = models.map(_.calculateRSquared())
val factorWeights = Array.ofDim[Double](stocksReturns.length, factors.length+1)
for (s <- 0 until stocksReturns.length) {
  factorWeights(s) = models(s).estimateRegressionParameters()
}
val factorCor = new PearsonsCorrelation(factorMat).getCorrelationMatrix().getData()
println(factorCor.map(_.mkString("\t")).mkString("\n"))
val factorCov = new Covariance(factorMat).getCovarianceMatrix().getData()
println(factorCov.map(_.mkString("\t")).mkString("\n"))
val factorMeans = factorsReturns.map(factor => factor.sum / factor.size)
val broadcastInstruments = sc.broadcast(factorWeights)
val parallelism = 1000
val baseSeed = 1496L
val seeds = (baseSeed until baseSeed + parallelism)
val seedRdd = sc.parallelize(seeds, parallelism)
val numTrials = 1
val trialValues = seedRdd.flatMap(trialReturns(_, numTrials / parallelism, broadcastInstruments.value, factorMeans, factorCov))
val topLosses = trialValues.takeOrdered(math.max(numTrials / 20, 1))
val varFivePercent = topLosses.last
val domain = Range.Double(20.0, 60.0, .2).toArray
val densities = KernelDensity.estimate(trialsRdd, 0.25, domain)
{% endhighlight %}

At any moment, the status of your cluster can be checked on the interface of the master node, on port 8080 :

![spark application interface]({{ site.url }}/img/spark_master.png)

which leads to status of each executor (you can check that memory is 6G out of 6.3G available) and to your client (localhost:4040) :

![spark application interface]({{ site.url }}/img/spark_client.png)

Be careful if you want to have multiple shell, you have to restrict the number of cores per shell using option in spark-shell :

    --total-executor-cores 4

such as : 

    ~/technologies/spark-1.4.1-bin-hadoop2.6/bin/spark-shell --master spark://ec2-52-19-187-119.eu-west-1.compute.amazonaws.com:7077 --driver-memory 1g --executor-memory 1g --driver-cores 1 --executor-cores 1 --total-executor-cores 4

#Geometric brownian motion

#Square-root diffusion

#Stochastic volatility
