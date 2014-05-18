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
# Downloads a (partial) wikipedia dump, trains and tests a classifier.
#
# To run:  change into the mahout directory and type:
# examples/bin/classify-wiki.sh

if [ "$1" = "--help" ] || [ "$1" = "--?" ]; then
  echo "This script Bayes and CBayes classifiers over the last wikipedia dump."
  exit
fi

# ensure that MAHOUT_HOME is set
if [[ -z "$MAHOUT_HOME" ]]; then
  echo "Please set MAHOUT_HOME."
  exit
fi

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then
  cd $SCRIPT_PATH
fi
START_PATH=`pwd`

if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
  HADOOP="$HADOOP_HOME/bin/hadoop"
  if [ ! -e $HADOOP ]; then
    echo "Can't find hadoop in $HADOOP, exiting"
    exit 1
  fi
fi

WORK_DIR=/tmp/mahout-work-wiki-${USER}
algorithm=( CBayes clean)
if [ -n "$1" ]; then
  choice=$1
else
  echo "Please select a number to choose the corresponding task to run"
  echo "1. ${algorithm[0]}"
  echo "2. ${algorithm[1]} -- cleans up the work area in $WORK_DIR"
  read -p "Enter your choice : " choice
fi

echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
alg=${algorithm[$choice-1]}

if [ "x$alg" != "xclean" ]; then
  echo "creating work directory at ${WORK_DIR}"

  mkdir -p ${WORK_DIR}
    if [ ! -e ${WORK_DIR}/wikixml ]; then
        mkdir -p ${WORK_DIR}/wikixml
        echo "Downloading wikipedia XML dump"        
        ########## partial small
         #curl http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p000000010p000010000.bz2 -o ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml.bz2        
         ########## partial larger - uncomment and cle
         curl http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles10.xml-p000925001p001325000.bz2 -o ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml.bz2
      
         ######### Uncomment for full wikipedia dump: 10G zipped
         #curl http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -o ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml.bz2
      
      echo "Extracting..."
       
      cd ${WORK_DIR}/wikixml && bunzip2 enwiki-latest-pages-articles.xml.bz2 && cd .. && cd ..
    fi

echo $START_PATH
#cd $START_PATH
#cd ../..

set -e

if [ "x$alg" == "xCBayes" ]; then

  set -x
  echo "Preparing wikipedia data"
  rm -rf ${WORK_DIR}/wiki
  mkdir ${WORK_DIR}/wiki
  cp $MAHOUT_HOME/examples/src/test/resources/country10.txt ${WORK_DIR}/country10.txt
  chmod 666 ${WORK_DIR}/country10.txt

  if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
    echo "Copying wikipedia data to HDFS"
    set +e
    $HADOOP dfs -rmr ${WORK_DIR}/wikixml
    set -e
    $HADOOP dfs -put ${WORK_DIR}/wikixml ${WORK_DIR}/wikixml
  fi

  echo "Creating sequence files from wikiXML"
  $MAHOUT_HOME/bin/mahout seqwiki -c ${WORK_DIR}/country10.txt -i ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml -o ${WORK_DIR}/wikipediainput

  echo "Converting sequence files to vectors using bi-grams"
  $MAHOUT_HOME/bin/mahout seq2sparse -i ${WORK_DIR}/wikipediainput -o ${WORK_DIR}/wikipedidaVecs -wt tfidf -lnorm -nv -ow -ng 2

  echo "Creating training and holdout set with a random 80-20 split of the generated vector dataset"
  $MAHOUT_HOME/bin/mahout split -i ${WORK_DIR}/wikipedidaVecs/tfidf-vectors/ --trainingOutput ${WORK_DIR}/training --testOutput ${WORK_DIR}/testing -rp 20 -ow -seq -xm sequential

  echo "Training Naive Bayes model"
  $MAHOUT_HOME/bin/mahout trainnb -i ${WORK_DIR}/training -el -o ${WORK_DIR}/model -li ${WORK_DIR}/labelindex -ow -c

  echo "Self testing on training set"
  $MAHOUT_HOME/bin/mahout testnb -i ${WORK_DIR}/training -m ${WORK_DIR}/model -l ${WORK_DIR}/labelindex -ow -o ${WORK_DIR}/output -c

  echo "Testing on holdout set: Bayes"
  $MAHOUT_HOME/bin/mahout testnb -i ${WORK_DIR}/testing -m ${WORK_DIR}/model -l ${WORK_DIR}/labelindex -ow -o ${WORK_DIR}/output -seq

 echo "Testing on holdout set: CBayes"
  $MAHOUT_HOME/bin/mahout testnb -i ${WORK_DIR}/testing -m ${WORK_DIR}/model -l ${WORK_DIR}/labelindex -ow -o ${WORK_DIR}/output  -c -seq
fi
elif [ "x$alg" == "xclean" ]; then
  rm -rf ${WORK_DIR}
  rm -rf /tmp/news-group.model
fi
# Remove the work directory
#
