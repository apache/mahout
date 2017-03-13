#!/bin/bash

# Starts Hadoop on Docker Container
/etc/bootstrap.sh
$HADOOP_HOME/bin/hdfs dfsadmin -safemode leave

$SPARK_HOME/sbin/start-all.sh

$HADOOP_HOME/bin/hdfs dfs -mkdir /data
$HADOOP_HOME/bin/hdfs dfs -put /usr/local/ml-latest-small/ratings.csv /data/ratings.csv

echo "Starting Spark Item Similarity Test (tests CLI Drivers)"
$MAHOUT_HOME/bin/mahout spark-itemsimilarity \
    --master spark://$HOSTNAME:7077 \
     --input /data/ratings.csv \
     --output /opt/mahout/it_output/spark_item_sim_output \
     --itemIDColumn 1 \
     --rowIDColumn 0 \
     --sparkExecutorMem 6g \
     >> $MAHOUT_HOME/dockerITs/logs/spark-item-similarity-$MASTER.out

$HADOOP_HOME/bin/hdfs dfs -get /opt/mahout/it_output/spark_item_sim_output $MAHOUT_HOME/dockerITs/output/spark_item_sim_output

echo "Testing Classify Wikipedia"
$MAHOUT_HOME/examples/bin/classify-wikipedia.sh -n 2


$HADOOP_HOME/bin/hdfs dfs -get /tmp/mahout-work-wiki/output $MAHOUT_HOME/dockerITs/output/wiki-output

$MAHOUT_HOME/bin/mahout spark-shell -i $MAHOUT_HOME/examples/bin/spark-document-classifier.mscala