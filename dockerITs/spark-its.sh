#!/bin/bash

# Starts Hadoop on Docker Container
/etc/bootstrap.sh
$HADOOP_HOME/bin/hdfs dfsadmin -safemode leave

$SPARK_HOME/sbin/start-all.sh

$HADOOP_HOME/bin/hdfs dfs -mkdir /data
$HADOOP_HOME/bin/hdfs dfs -put /usr/local/ml-latest-small/ratings.csv /data/ratings.csv

echo "Starting Spark Item Similarity Test"
$MAHOUT_HOME/bin/mahout spark-itemsimilarity \
    --master spark://$HOSTNAME:7077 \
     --input /data/ratings.csv \
     --output /opt/mahout/it_output/spark_item_sim_output \
     --itemIDColumn 1 \
     --rowIDColumn 0 \
     --sparkExecutorMem 6g
     >> $MAHOUT_HOME/dockerITs/logs/spark-item-similarity-$MASTER.out