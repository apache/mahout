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
# Downloads the 20newsgroups dataset, trains and tests a classifier.
#
# To run:  change into the mahout directory and type:
#  examples/bin/build-20news.sh

if [ "$1" = "--help" ] || [ "$1" = "--?" ]; then
  echo "This script runs SGD and Bayes classifiers over the classic 20 News Groups."
  exit
fi

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then
  cd $SCRIPT_PATH
fi
START_PATH=`pwd`

WORK_DIR=/tmp/mahout-work-${USER}
algorithm=( cnaivebayes naivebayes sgd clean)
if [ -n "$1" ]; then
  choice=$1
else
  echo "Please select a number to choose the corresponding task to run"
  echo "1. ${algorithm[0]}"
  echo "2. ${algorithm[1]}"
  echo "3. ${algorithm[2]}"
  echo "4. ${algorithm[3]} -- cleans up the work area in $WORK_DIR"
  read -p "Enter your choice : " choice
fi

echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
alg=${algorithm[$choice-1]}

echo "creating work directory at ${WORK_DIR}"

mkdir -p ${WORK_DIR}
if [ ! -e ${WORK_DIR}/20news-bayesinput ]; then
  if [ ! -e ${WORK_DIR}/20news-bydate ]; then
    if [ ! -f ${WORK_DIR}/20news-bydate.tar.gz ]; then
      echo "Downloading 20news-bydate"
      curl http://people.csail.mit.edu/jrennie/20Newsgroups/20news-bydate.tar.gz -o ${WORK_DIR}/20news-bydate.tar.gz
    fi
    mkdir -p ${WORK_DIR}/20news-bydate
    echo "Extracting..."
    cd ${WORK_DIR}/20news-bydate && tar xzf ../20news-bydate.tar.gz && cd .. && cd ..
  fi
fi
#echo $START_PATH
cd $START_PATH
cd ../..

set -e

if [ "x$alg" == "xnaivebayes"  -o  "x$alg" == "xcnaivebayes" ]; then
  c=""

  if [ "x$alg" == "xcnaivebayes" ]; then
    c=" -c"
  fi

  set -x
  echo "Preparing 20newsgroups data"
  rm -rf ${WORK_DIR}/20news-all
  mkdir ${WORK_DIR}/20news-all
  cp -R ${WORK_DIR}/20news-bydate/*/* ${WORK_DIR}/20news-all

  echo "Creating sequence files from 20newsgroups data"
  ./bin/mahout seqdirectory \
    -i ${WORK_DIR}/20news-all \
    -o ${WORK_DIR}/20news-seq

  echo "Converting sequence files to vectors"
  ./bin/mahout seq2sparse \
    -i ${WORK_DIR}/20news-seq \
    -o ${WORK_DIR}/20news-vectors  -lnorm -nv  -wt tfidf

  echo "Creating training and holdout set with a random 80-20 split of the generated vector dataset"
  ./bin/mahout split \
    -i ${WORK_DIR}/20news-vectors/tfidf-vectors \
    --trainingOutput ${WORK_DIR}/20news-train-vectors \
    --testOutput ${WORK_DIR}/20news-test-vectors  \
    --randomSelectionPct 40 --overwrite --sequenceFiles -xm sequential

  echo "Training Naive Bayes model"
  ./bin/mahout trainnb \
    -i ${WORK_DIR}/20news-train-vectors -el \
    -o ${WORK_DIR}/model \
    -li ${WORK_DIR}/labelindex \
    -ow $c

  echo "Self testing on training set"

  ./bin/mahout testnb \
    -i ${WORK_DIR}/20news-train-vectors\
    -m ${WORK_DIR}/model \
    -l ${WORK_DIR}/labelindex \
    -ow -o ${WORK_DIR}/20news-testing $c

  echo "Testing on holdout set"

  ./bin/mahout testnb \
    -i ${WORK_DIR}/20news-test-vectors\
    -m ${WORK_DIR}/model \
    -l ${WORK_DIR}/labelindex \
    -ow -o ${WORK_DIR}/20news-testing $c

elif [ "x$alg" == "xsgd" ]; then
  if [ ! -e "/tmp/news-group.model" ]; then
    echo "Training on ${WORK_DIR}/20news-bydate/20news-bydate-train/"
    ./bin/mahout org.apache.mahout.classifier.sgd.TrainNewsGroups ${WORK_DIR}/20news-bydate/20news-bydate-train/
  fi
  echo "Testing on ${WORK_DIR}/20news-bydate/20news-bydate-test/ with model: /tmp/news-group.model"
  ./bin/mahout org.apache.mahout.classifier.sgd.TestNewsGroups --input ${WORK_DIR}/20news-bydate/20news-bydate-test/ --model /tmp/news-group.model
elif [ "x$alg" == "xclean" ]; then
  rm -rf ${WORK_DIR}
  rm -rf /tmp/news-group.model
fi
# Remove the work directory
#
