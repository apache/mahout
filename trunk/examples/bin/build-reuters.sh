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

cd examples/bin/
mkdir -p work
if [ ! -e work/reuters-out ]; then
  if [ ! -e work/reuters-sgm ]; then
    if [ ! -f work/reuters21578.tar.gz ]; then
      echo "Downloading Reuters-21578"
      curl http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz  -o work/reuters21578.tar.gz
    fi
    mkdir -p work/reuters-sgm
    echo "Extracting..."
    cd work/reuters-sgm && tar xzf ../reuters21578.tar.gz && cd .. && cd ..
  fi
fi
cd ../..

./bin/mahout org.apache.lucene.benchmark.utils.ExtractReuters \
  ./examples/bin/work/reuters-sgm/ \
  ./examples/bin/work/reuters-out/ \
&& \
./bin/mahout seqdirectory \
  -i ./examples/bin/work/reuters-out/ \
  -o ./examples/bin/work/reuters-out-seqdir \
  -c UTF-8 -chunk 5

if [ "x$clustertype" == "xkmeans" ]; then
  ./bin/mahout seq2sparse \
    -i ./examples/bin/work/reuters-out-seqdir/ \
    -o ./examples/bin/work/reuters-out-seqdir-sparse \
  && \
  ./bin/mahout kmeans \
    -i ./examples/bin/work/reuters-out-seqdir-sparse/tfidf-vectors/ \
    -c ./examples/bin/work/clusters \
    -o ./examples/bin/work/reuters-kmeans \
    -x 10 -k 20 -ow \
  && \
  ./bin/mahout clusterdump \
    -s examples/bin/work/reuters-kmeans/clusters-10 \
    -d examples/bin/work/reuters-out-seqdir-sparse/dictionary.file-0 \
    -dt sequencefile -b 100 -n 20
elif [ "x$clustertype" == "xlda" ]; then
  ./bin/mahout seq2sparse \
    -i ./examples/bin/work/reuters-out-seqdir/ \
    -o ./examples/bin/work/reuters-out-seqdir-sparse \
    -wt tf -seq -nr 3 \
  && \
  ./bin/mahout lda \
    -i ./examples/bin/work/reuters-out-seqdir-sparse/tf-vectors \
    -o ./examples/bin/work/reuters-lda -k 20 -v 50000 -ow -x 20 \
  && \
  ./bin/mahout ldatopics \
    -i ./examples/bin/work/reuters-lda/state-20 \
    -d ./examples/bin/work/reuters-out-seqdir-sparse/dictionary.file-0 \
    -dt sequencefile
else 
  echo "unknown cluster type: $clustertype";
fi 

