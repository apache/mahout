#!/bin/sh
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
# Downloads the 20newsgroups dataset, trains and tests a bayes classifier. 
#
# To run:  change into the mahout directory and type:
#  examples/bin/build-20news.sh

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then 
  cd $SCRIPT_PATH
fi

mkdir -p work
if [ ! -e work/20news-bayesinput ]; then
  if [ ! -e work/20news-bydate ]; then
    if [ ! -f work/20news-bydate.tar.gz ]; then
      echo "Downloading 20news-bydate"
      curl http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz -o work/20news-bydate.tar.gz
    fi
    mkdir -p work/20news-bydate
    echo "Extracting..."
    cd work/20news-bydate && tar xzf ../20news-bydate.tar.gz && cd .. && cd ..
  fi
fi

cd ../..

set -e

./bin/mahout org.apache.mahout.classifier.bayes.PrepareTwentyNewsgroups \
  -p examples/bin/work/20news-bydate/20news-bydate-train \
  -o examples/bin/work/20news-bydate/bayes-train-input \
  -a org.apache.mahout.vectorizer.DefaultAnalyzer \
  -c UTF-8

./bin/mahout org.apache.mahout.classifier.bayes.PrepareTwentyNewsgroups \
  -p examples/bin/work/20news-bydate/20news-bydate-test \
  -o examples/bin/work/20news-bydate/bayes-test-input \
  -a org.apache.mahout.vectorizer.DefaultAnalyzer \
  -c UTF-8 

TEST_METHOD="sequential"

# if we're set up to run on a cluster..
if [ "$HADOOP_HOME" != "" ]; then
    # mapreduce test method used on hadoop
    TEST_METHOD="mapreduce"

    set +e 
    hadoop dfs -rmr \
      examples/bin/work/20news-bydate/bayes-train-input 

    hadoop dfs -rmr \
      examples/bin/work/20news-bydate/bayes-test-input

    set -e
    hadoop dfs -put \
      examples/bin/work/20news-bydate/bayes-train-input \
      examples/bin/work/20news-bydate/bayes-train-input 

    hadoop dfs -put \
      examples/bin/work/20news-bydate/bayes-test-input \
      examples/bin/work/20news-bydate/bayes-test-input
fi


./bin/mahout trainclassifier \
  -i examples/bin/work/20news-bydate/bayes-train-input \
  -o examples/bin/work/20news-bydate/bayes-model \
  -type bayes \
  -ng 1 \
  -source hdfs

./bin/mahout testclassifier \
  -m examples/bin/work/20news-bydate/bayes-model \
  -d examples/bin/work/20news-bydate/bayes-test-input \
  -type bayes \
  -ng 1 \
  -source hdfs \
  -method ${TEST_METHOD}
