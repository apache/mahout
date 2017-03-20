#!/bin/bash

# Starts Hadoop on Docker Container
/etc/bootstrap.sh
$HADOOP_HOME/bin/hdfs dfsadmin -safemode leave

$SPARK_HOME/sbin/start-all.sh

# Create directory for test outputs
rm -r $MAHOUT_HOME/tmp
echo "Creating workdir"
mkdir -p $MAHOUT_HOME/tmp/logs

cp $MAHOUT_HOME/dockerITs/log4j.properties $SPARK_HOME/conf/log4j.properties

# Create data directory on HDFS
$HADOOP_HOME/bin/hdfs dfs -mkdir /data
$HADOOP_HOME/bin/hdfs dfs -put /usr/local/ml-latest-small/ratings.csv /data/ratings.csv

export MASTER=spark://$HOSTNAME:7077
echo "Master set at ${MASTER}"

echo "Starting Spark Item Similarity Test (tests CLI Drivers)"
$MAHOUT_HOME/bin/mahout spark-itemsimilarity \
    --master spark://$HOSTNAME:7077 \
     --input /data/ratings.csv \
     --output /tmp/mahout/spark_item_sim_output \
     --itemIDColumn 1 \
     --rowIDColumn 0 \
     --sparkExecutorMem 6g \
     -D:spark.eventLog.dir="$MAHOUT_HOME/tmp/logs"
     >> $MAHOUT_HOME/tmp/spark-item-similarity-test.out

$HADOOP_HOME/bin/hdfs dfs -get /tmp/mahout/spark_item_sim_output $MAHOUT_HOME/tmp/spark_item_sim_output
#
echo "Testing Classify Wikipedia"
## This is actually a Map Reduce Job
$MAHOUT_HOME/examples/bin/classify-wikipedia.sh 2 >> $MAHOUT_HOME/tmp/classify-wikipedia_2.out
echo $HADOOP_HOME/bin/hdfs dfs -ls /tmp/mahout-work-wiki/output
$HADOOP_HOME/bin/hdfs dfs -get /tmp/mahout-work-wiki/output $MAHOUT_HOME/tmp/wiki-output

$MAHOUT_HOME/bin/mahout spark-shell -i $MAHOUT_HOME/examples/bin/spark-document-classifier.mscala >> spark-document-classifier.out