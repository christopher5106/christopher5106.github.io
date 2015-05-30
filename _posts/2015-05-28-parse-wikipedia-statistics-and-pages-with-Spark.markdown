---
layout: post
title:  "Parse Wikipedia statistics and pages with Spark"
date:   2015-05-28 23:00:51
categories: bigdata
---

#Launch your Spark cluster on EC2

Let's launch a cluster of 5 AWS EC2 instances (1 master and 4 slaves) of type m1.large with Spark.

To prepare

- download the latest version of [Spark](https://spark.apache.org/downloads.html)

{% highlight bash %}
wget http://apache.websitebeheerjd.nl/spark/spark-1.3.1/spark-1.3.1-bin-hadoop2.6.tgz
tar xvf spark-1.3.1-bin-hadoop2.6.tgz
{% endhighlight %}

- create an AWS account, and get your credentials

- create an EC2 key pair named `sparkclusterkey`

{% highlight bash %}
#download the keypair and change the rights
mv Downloads/sparkclusterkey.pem.txt sparkclusterkey.pem
chmod 600 sparkclusterkey.pem
{% endhighlight %}

Launch the [cluster](https://spark.apache.org/docs/1.2.0/ec2-scripts.html) :

{% highlight bash %}
#export the AWS credentials
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

#launch the cluster with --copy-aws-credentials option to enable S3 access.
./spark-1.3.1-bin-hadoop2.6/ec2/spark-ec2 -k sparkclusterkey -i sparkclusterkey.pem --region=eu-west-1 --copy-aws-credentials --instance-type=m1.large -s 4 --hadoop-major-version=2 launch spark-cluster

#connect to the master
./spark-1.3.1-bin-hadoop2.6/ec2/spark-ec2 -k sparkclusterkey -i sparkclusterkey.pem --region=eu-west-1 login spark-cluster

#launch the shell
./spark/bin/spark-shell
{% endhighlight %}

To persist the data when you close the cluster, you can add for example an EBS of 30G to each instance with option `--ebs-vol-size=30`, if the data you need to persist will require less than 150GB (5 x 30). You'll also need to change the HDFS for persistent (see below).

Spark master web interface will be available on the master node on port `8080`.

Spark master web interface will be available on the master node on port `4040`.

**You're now ready !**

#Analyze the top ranking pages from October 2008 to February 2010 with Wikipedia statistics

Let's use the [dataset prepared by AWS](http://aws.amazon.com/datasets/4182) where lines are looking like :

    en Barack_Obama 997 123091092
    en Barack_Obama%27s_first_100_days 8 850127
    en Barack_Obama,_Jr 1 144103
    en Barack_Obama,_Sr. 37 938821
    en Barack_Obama_%22HOPE%22_poster 4 81005
    en Barack_Obama_%22Hope%22_poster 5 102081

and launch the [associated example](https://aws.amazon.com/articles/4926593393724923)

{% highlight scala %}
val file = sc.textFile("s3n://support.elasticmapreduce/bigdatademo/sample/wiki")
val reducedList = file.map(l => l.split(" ")).map(l => (l(1), l(2).toInt)).reduceByKey(_+_, 3)
reducedList.cache
val sortedList = reducedList.map(x => (x._2, x._1)).sortByKey(false).take(50)
{% endhighlight %}


Be careful to change "s3" on "s3**n**".

**That's it! 63 seconds to get all the data ranked !**

#Analyze the principal concepts in the Wikipedia pages

By default, it's the ephemeral HDFS that has been started, with 3.43 TB capacity and  a web interface available on port `50070`.

{% highlight bash %}
#have a look if everything works fine
./ephemeral-hdfs/bin/hadoop fs -ls /
{% endhighlight %}

Persistent HDFS give by default 31.5 GB of space which is small. If you have not choosen to add the EBS option, persistent HDFS is not worth. If you did choose the EBS option, I believe you choose the EBS capacity in function of your needs and you can switch for persistent HDFS to be able to stop the cluster to save money while keeping the data in a persistent way :

{% highlight bash %}
#stop ephemeral hdfs
./ephemeral-hdfs/bin/stop-dfs.sh

#start persistent hdfs
./persistent-hdfs/bin/start-dfs.sh
{% endhighlight %}

Permanent HDFS web interface will be available on port `60070` by default.

Depending on your choice, export the path to ephermeral or persistent HDFS :

{% highlight bash %}
export PATH=$PATH:./ephemeral-hdfs/bin
{% endhighlight %}

Let's download the [French Wikipedia database](http://dumps.wikimedia.org/frwiki/20150512/) and the English Wikipedia database.

{% highlight bash %}
#download may take a while
curl -s -L http://dumps.wikimedia.org/frwiki/20150512/frwiki-20150512-pages-articles-multistream.xml.bz2 | bzip2 -cd | hadoop fs -put - /user/ds/wikidump.xml
curl -s -L http://dumps.wikimedia.org/enwiki/20150304/enwiki-20150304-pages-articles-multistream.xml.bz2 | bzip2 -cd | hadoop fs -put - /user/ds/wikidump-en.xml
{% endhighlight %}

The FR database, of size 12.8GB, is divided into 103 blocks, replicated 3 times, using then 38.63GB of our 3.43 TB of total capacity for the cluster, hence around 10GB of each datanode of 826GB capacity.
The EN database, of size 48.4GB, is divided into 388 blocks replicated 3 times. The data represents 7% of the cluster capacity, which is fine.

|  | FR wiki |  --    EN wiki    --  | Available |
| ------------- | ------------- | ------------- | ------------- |
| Size per Cluster  |     38.63GB | 140GB | 3.43 TB |
| Size per Node  | 10GB  | 35GB  | 826GB |
| Number of Blocks | 103 | 388 | -- |


Now it's time to [study our database](https://github.com/sryza/aas) and extract its principal concepts with MLlib singular value decomposition following ["Advanced Analytics with Spark" example](http://shop.oreilly.com/product/0636920035091.do) :

{% highlight bash %}
#download code (revision hash eed30d7214e0e7996068083f2ac6793e6375a768)
git clone https://github.com/sryza/aas.git

#install maven
wget http://mirror.olnevhost.net/pub/apache/maven/maven-3/3.0.5/binaries/apache-maven-3.0.5-bin.tar.gz
tar xvf apache-maven-3.0.5-bin.tar.gz
mv apache-maven-3.0.5  /usr/local/apache-maven
export M2_HOME=/usr/local/apache-maven
export M2=$M2_HOME/bin
export PATH=$M2:$PATH

#check the versions and edit pom.xml properties
hadoop version
java -version
mvn -version
scala -version
./spark/bin/spark-submit --version
vi aas/pom.xml #I kept Hadoop version 2.6.0 in order to compile

#compile
cd aas
mvn install #or mvn install -pl common
cd ch06-lsa
mvn clean && mvn compile && mvn package
cd ../..
{% endhighlight %}

You can either submit the program

{% highlight bash %}
#upload 'stopwords.txt' to 'hdfs:///user/ds/stopwords.txt' and change the path in RunLSA.scala file, don't forget to recompile
./ephemeral-hdfs/bin/hadoop fs -cp file:///root/aas/ch06-lsa/src/main/resources/stopwords.txt hdfs:///user/ds/
vi aas/ch06-lsa/src/main/scala/com/cloudera/datascience/lsa/RunLSA.scala

#submit the job
./spark/bin/spark-submit --class com.cloudera.datascience.lsa.RunLSA aas/ch06-lsa/target/ch06-lsa-1.0.0-jar-with-dependencies.jar
{% endhighlight %}

or test it step by step with the shell

{% highlight bash %}
#launch spark shell
./spark/bin/spark-shell --jars aas/ch06-lsa/target/ch06-lsa-1.0.0-jar-with-dependencies.jar --driver-memory 6g
{% endhighlight %}

and check if everything works well, in particular reading the files

{% highlight scala %}
/*test reading the xml file*/
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io._
@transient val conf = new Configuration()
import com.cloudera.datascience.common.XmlInputFormat
conf.set(XmlInputFormat.START_TAG_KEY, "<page>")
conf.set(XmlInputFormat.END_TAG_KEY, "</page>")
import org.apache.hadoop.io.{LongWritable, Text}
val rawXmls = sc.newAPIHadoopFile("hdfs:///user/ds/wikidump.xml", classOf[XmlInputFormat],classOf[LongWritable],classOf[Text], conf)

/*test the sampling*/
val allpages = rawXmls.map(p => p._2.toString)
val pages = allpages.sample(false, 0.1, 11L)

/*test the parsing*/
import com.cloudera.datascience.lsa.ParseWikipedia._
val plainText = pages.filter(_ != null).flatMap(wikiXmlToPlainText)

/*test the broadcast*/
val stopWords = sc.broadcast(loadStopWords("aas/ch06-lsa/src/main/resources/stopwords.txt")).value

/*test the stemming*/  
val lemmatized = plainText.mapPartitions(iter => {
  val pipeline = createNLPPipeline()
  iter.map{ case(title, contents) => (title, plainTextToLemmas(contents, stopWords, pipeline))}
})
val filtered = lemmatized.filter(_._2.size > 1)

/*test the creation of the term-document matrix*/
val (termDocMatrix, termIds, docIds, idfs) = termDocumentMatrix(filtered, stopWords, 100000, sc)
termDocMatrix.cache()

/*extract 100 concepts*/
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val mat = new RowMatrix(termDocMatrix)
val svd = mat.computeSVD(100, computeU=true)

println("Singular values: " + svd.s)
val topConceptTerms = topTermsInTopConcepts(svd, 10, 10, termIds)
val topConceptDocs = topDocsInTopConcepts(svd, 10, 10, docIds)
for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
  println("Concept terms: " + terms.map(_._1).mkString(", "))
  println("Concept docs: " + docs.map(_._1).mkString(", "))
  println()
}

{% endhighlight %}

#Index Wikipedia pages with Elasticsearch to search them geographically

I'll begin with a minimal mapping, in particular to avoid dynamic mapping to match wrong types, `mapping.json` :

{% highlight json %}
{
    "poi" : {
      "properties": {
        "infobox": {"type":"string"},
        "category": {"type":"string"},
        "name": {"type":"string"},
        "image": {"type":"string"},
        "description": {"type":"string"},
        "CodeSkip": {"type":"string"},
        "location": {"type":"geo_point"}
      }
    }
}
{% endhighlight %}


Once Elasticsearch installed, let's create an index and an alias `map` so that we can create multiple index behind...

{% highlight bash %}
curl -XPUT 52.17.250.224:9200/map1
curl -XPUT 52.17.250.224:9200/map1/poi/_mapping -d @mapping.json
curl -XPUT 52.17.250.224:9200/map2
curl -XPUT 52.17.250.224:9200/map2/poi/_mapping -d @mapping.json
curl -XPOST 'http://52.17.250.224:9200/_aliases' -d '
{
    "actions" : [
        { "add" : { "index" : "map1", "alias" : "map" } }
    ]
}'
#check the aliases and mappings
curl -XGET 'http://52.17.250.224:9200/_mapping'
curl -XGET 'http://52.17.250.224:9200/_aliases'
{% endhighlight %}

Let's download a Wikipedia XML API and launch Spark Shell :

{% highlight bash %}

wget
http://central.maven.org/maven2/org/elasticsearch/elasticsearch-spark_2.10/2.1.0.Beta2/elasticsearch-spark_2.10-2.1.0.Beta2.jar

./spark/bin/spark-shell --jars aas/ch06-lsa/target/ch06-lsa-1.0.0-jar-with-dependencies.jar,elasticsearch-spark_2.10-2.1.0.Beta2.jar
{% endhighlight %}

and parse the data, filter pages with images and coordinates, and send to Elastichsearch for bulk indexation

{% highlight scala %}
//let's open again
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io._
@transient val conf = new Configuration()
import com.cloudera.datascience.common.XmlInputFormat
conf.set(XmlInputFormat.START_TAG_KEY, "<page>")
conf.set(XmlInputFormat.END_TAG_KEY, "</page>")
import org.apache.hadoop.io.{LongWritable, Text}
val rawXmls = sc.newAPIHadoopFile("hdfs:///user/ds/wikidump_fr.xml", classOf[XmlInputFormat],classOf[LongWritable],classOf[Text], conf)
rawXmls.cache();

//let's parse the pages and infoboxes
import com.cloudera.datascience.lsa.ParseWikipedia._
val pois = rawXmls.filter(_ != null).map(p => {
  val onepage = p._2.toString
  val splitArray = onepage.split("\n");
  // parse infobox
  val map = collection.mutable.Map[String, String]()
  var init = false;
  for(line <- splitArray) {
    if(!init && ( line.startsWith("{infobox",1) || line.startsWith("{Infobox",1)) ) {
      map += ("infobox" -> line.span(_ != ' ')._2.trim.replace("\"", "").toLowerCase() )
      init = true ;
    }
    else
      if (init && line.startsWith("|") ) {
        val lineSplits = line.span(_ != '=');
        val key = lineSplits._1.stripPrefix("|").trim.replace("\"", "").toLowerCase()
        val value = lineSplits._2.stripPrefix("=").trim.replace("\"", "").toLowerCase()
        if(value != "") {
          map += (key -> value )
          if( Seq("lat","lat1","latitude1") contains key )
              map += ("latitude" -> value )
          if( Seq("lon","lon1","longitude1", "long") contains key )
              map += ("longitude" -> value )
          if( Seq("photo") contains key )
            map += ("image" -> value)
        }
      }
      else if( init && line.startsWith("}}") )
        init = false;
  }
  //plain text
  val plain = wikiXmlToPlainText(onepage)
  if(!plain.isEmpty)
    map += ("description" ->  plain.get._2.replace("\n", ""), "title" -> plain.get._1)
  map
})
pois.cache()


val filtered = pois.filter( map => map.contains("latitude") && map.contains("longitude") && map.contains("image"))

val patR3 = """(-*\d*.\d*)""".r
val filtered2 = filtered.filter( x => {
  (x("latitude"),x("longitude")) match {  
    case (patR3(i),patR3(j)) => true ;
    case i => false
  }
})

//get location
val poisWithLocation = filtered2.map( map => map + ( "location" -> ( map("latitude") + ", " + map("longitude") ) ) )

//let's index in Elasticsearch
import org.elasticsearch.spark._
poisWithLocation.saveToEs("map2/poi", Map("es.nodes" -> "52.17.250.224","index.mapping.ignore_malformed" -> "true"))


{% endhighlight %}



Let's see what kind of infobox we have and how many are geo localized :

{% highlight bash %}
curl -XGET 'http://52.17.250.224:9200/map2/poi/_search?search_type=count&pretty' -d '{
 "aggregations": {
   "infoboxRepartition": {
     "terms": {
       "field": "infobox",
       "order": {
                "_count" : "desc"
              }
     }
   },
   "number_of_location": {
     "value_count": {
       "field":"location"
     }
   }
 }
}
'
{% endhighlight %}

We can see that we have xxx communes, xxx france, ...

To find the relevant points of interest around Paris :

{% highlight bash %}
curl -XGET http://52.17.250.224:9200/map2/poi/_search -d '{
  "query":{
    "match_all":{}
  },
  "sort": [{
    "_geo_distance": {
      "location":"48.8567, 2.3508",
      "unit":"km"
    }
  }]
}'
{% endhighlight %}


#Stop, restart or destroy the cluster

{% highlight bash %}
#stop
./spark-1.3.1-bin-hadoop2.6/spark-ec2 -k sparkclusterkey -i sparkclusterkey.pem --region=eu-west-1  stop spark-cluster
#restart
./spark-1.3.1-bin-hadoop2.6/spark-ec2 -k sparkclusterkey -i sparkclusterkey.pem --region=eu-west-1  start spark-cluster
#destroy
./spark-1.3.1-bin-hadoop2.6/spark-ec2 -k sparkclusterkey -i sparkclusterkey.pem --region=eu-west-1  destroy spark-cluster
{% endhighlight %}

**Well done !**
