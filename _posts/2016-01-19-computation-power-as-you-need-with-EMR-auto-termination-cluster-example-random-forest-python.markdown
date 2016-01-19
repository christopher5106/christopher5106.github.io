---
layout: post
title:  "Computation power as you need with EMR auto-terminating clusters : example for a random forest evaluation in Python"
date:   2016-01-19 23:00:51
categories: big data
---

One of the main advantage of the cloud is to have the possibility to rent a *temporary* computation power, for a short period of time.

With **auto-terminating EMR cluster**, it is also possible to use a cluster periodically, for example every month, for a specific big data task, such as updating prediction models from the production.

The cost of using a cluster of **100** *m3.xlarge* instances ($0.266 per hour per instance) for **1 hour** will be **$26**.

Let's see in practice with the computation of a Random Forest regressor. Computing a Random Forest regressor (RFR) on a high volume of data will require a few days, which is not acceptable for R&D as well as for production : a failure in the process would postpone the update by 5 days. Let's see how to launch a cluster every month.

In the following case, preprocessed production data will be a 5.6 G CSV file, where each column is separated by a `;` character, the first column corresponding to the label to predict, and the following columns the data to use for the prediction in the regressor.

To parallelize on 100 (one hundred) AWS EC2 instances, AWS first requires to **raise the current account's EC2 limit** with [a form](https://aws.amazon.com/support/createCase?type=service_limit_increase&serviceLimitIncreaseType=ec2-instances).

Computing a RFR on a cluster with Spark is as simple as with other libraries.

I'll create a python **compute_rf.py** file :

{% highlight python %}
from pyspark import SparkConf, SparkContext
sc = SparkContext()

import sys
filename = sys.argv[1]
print "Opening " + filename
file = sc.textFile(filename)

from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

#get header line, and column indexes
headers = file.take(1)[0].split(";")
index_label = headers.index("label")
index_expl_start = headers.index('expl1')
index_expl_stop = len(headers)

#transform line to LabeledPoint
def transform_line_to_labeledpoint(line):
    values = line.split(";")
    if values[index_label] == "":
        label = 0.0
    else:
        label = float(values[index_label])
    vector = []
    for i in range(index_expl_start, index_expl_stop):
        if values[i] == "":
            vector.append(0.0)
        else:
            vector.append(float(values[i]))
    return LabeledPoint(label,vector)

#filter first line, and transform input to a LabeledPoint RDD
data = file.filter(lambda w: not w.startswith(";xm;ym;anpol;numcnt;")).map(transform_line_to_labeledpoint)

(trainingData, testData) = data.randomSplit([0.7, 0.3])

trainingData.cache()
testData.cache()

model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},numTrees=2500, featureSubsetStrategy="sqrt",impurity='variance')
# does not work : ,minInstancesPerNode=1000)

predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression forest model:')
print(model.toDebugString())

{% endhighlight %}


A a third step, **upload to AWS S3** the data CSV file and the python file *compute_rf.py*.

Last step, launch the cluster with an additional step :

![EMR additional step]({{ site.url }}/img/emr_add_step.png)

Select **auto-terminating** option to close the cluster automatically once the computation is done :

![EMR auto terminate]({{ site.url }}/img/emr_auto_termination.png)

Choose the number of instances :

![EMR instances]({{ site.url }}/img/emr_power.png)

And define where to output the logs :

![EMR logs]({{ site.url }}/img/emr_output.png)
