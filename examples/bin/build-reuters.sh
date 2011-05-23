#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# Downloads the Reuters dataset and prepares it for clustering
#
# To run:  change into the mahout directory and type:
#  examples/bin/build-reuters.sh

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then 
  cd $SCRIPT_PATH
fi

MAHOUT="../../bin/mahout"

if [ ! -e $MAHOUT ]; then
  echo "Can't find mahout driver in $MAHOUT, cwd `pwd`, exiting.."
  exit 1
fi

if [ "$1" = "-ni" ]; then
  clustertype=kmeans
else
  algorithm=( kmeans lda )
 
  echo "Please select a number to choose the corresponding clustering algorithm"
  echo "1. ${algorithm[0]} clustering"
  echo "2. ${algorithm[1]} clustering"
  read -p "Enter your choice : " choice

  echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]} Clustering"
  clustertype=${algorithm[$choice-1]} 
fi

mkdir -p mahout-work

if [ ! -e mahout-work/reuters-out-seqdir ]; then
    if [ ! -e mahout-work/reuters-out ]; then
	if [ ! -e mahout-work/reuters-sgm ]; then
	    if [ ! -f mahout-work/reuters21578.tar.gz ]; then
		echo "Downloading Reuters-21578"
		curl http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz \
                     -o mahout-work/reuters21578.tar.gz
	    fi
	    mkdir -p mahout-work/reuters-sgm
	    echo "Extracting..."
	    cd mahout-work/reuters-sgm && tar xzf ../reuters21578.tar.gz && cd .. && cd ..
	fi
	
	$MAHOUT org.apache.lucene.benchmark.utils.ExtractReuters \
	    mahout-work/reuters-sgm \
	    mahout-work/reuters-out 
    fi

    MAHOUT_LOCAL=true $MAHOUT seqdirectory \
        -i mahout-work/reuters-out \
        -o mahout-work/reuters-out-seqdir \
        -c UTF-8 -chunk 5
fi

# we know reuters-out-seqdir exists on a local disk at
# this point, if we're running in clustered mode, 
# copy it up to hdfs
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
    HADOOP="$HADOOP_HOME/bin/hadoop"
    if [ ! -e $HADOOP ]; then 
	echo "Can't find hadoop in $HADOOP, exiting"
	exit 1
    fi

    set +e
    $HADOOP dfs -rmr \
        mahout-work/reuters-out-seqdir
    set -e
    $HADOOP dfs -put \
        mahout-work/reuters-out-seqdir \
        mahout-work/reuters-out-seqdir
fi

if [ "x$clustertype" == "xkmeans" ]; then
  $MAHOUT seq2sparse \
    -i mahout-work/reuters-out-seqdir/ \
    -o mahout-work/reuters-out-seqdir-sparse-kmeans \
  && \
  $MAHOUT kmeans \
    -i mahout-work/reuters-out-seqdir-sparse-kmeans/tfidf-vectors/ \
    -c mahout-work/reuters-kmeans-clusters \
    -o mahout-work/reuters-kmeans \
    -x 10 -k 20 -ow \
  && \
  $MAHOUT clusterdump \
    -s mahout-work/reuters-kmeans/clusters-10 \
    -d mahout-work/reuters-out-seqdir-sparse-kmeans/dictionary.file-0 \
    -dt sequencefile -b 100 -n 20
elif [ "x$clustertype" == "xlda" ]; then
  $MAHOUT seq2sparse \
    -i mahout-work/reuters-out-seqdir/ \
    -o mahout-work/reuters-out-seqdir-sparse-lda \
    -wt tf -seq -nr 3 \
  && \
  $MAHOUT lda \
    -i mahout-work/reuters-out-seqdir-sparse-lda/tf-vectors \
    -o mahout-work/reuters-lda -k 20 -v 50000 -ow -x 20 \
  && \
  $MAHOUT ldatopics \
    -i mahout-work/reuters-lda/state-20 \
    -d mahout-work/reuters-out-seqdir-sparse-lda/dictionary.file-0 \
    -dt sequencefile
else 
  echo "unknown cluster type: $clustertype";
fi 
