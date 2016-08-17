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
# examples/bin/classify-wikipedia.sh

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

# Set commands for dfs
source ${START_PATH}/set-dfs-commands.sh

if [[ -z "$MAHOUT_WORK_DIR" ]]; then
  WORK_DIR=/tmp/mahout-work-wiki
else
  WORK_DIR=$MAHOUT_WORK_DIR
fi
algorithm=( CBayes BinaryCBayes clean)
if [ -n "$1" ]; then
  choice=$1
else
  echo "Please select a number to choose the corresponding task to run"
  echo "1. ${algorithm[0]} (may require increased heap space on yarn)"
  echo "2. ${algorithm[1]}"
  echo "3. ${algorithm[2]} -- cleans up the work area in $WORK_DIR"
  read -p "Enter your choice : " choice
fi

echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
alg=${algorithm[$choice-1]}

if [ "x$alg" != "xclean" ]; then
  echo "creating work directory at ${WORK_DIR}"

  mkdir -p ${WORK_DIR}
    if [ ! -e ${WORK_DIR}/wikixml ]; then
        mkdir -p ${WORK_DIR}/wikixml
    fi
    if [ ! -e ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml.bz2 ]; then
        echo "Downloading wikipedia XML dump"
        ########################################################   
        #  Datasets: uncomment and run "clean" to change dataset   
        ########################################################
        ########## partial small 42.5M zipped
        # curl https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p000000010p000030302.bz2 -o ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml.bz2
        ########## partial larger 256M zipped
        curl https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles10.xml-p002336425p003046511.bz2 -o ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml.bz2
        ######### full wikipedia dump: 10G zipped
        # curl https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -o ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml.bz2
        ########################################################
    fi
    if [ ! -e ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml ]; then
        echo "Extracting..."
       
        cd ${WORK_DIR}/wikixml && bunzip2 enwiki-latest-pages-articles.xml.bz2 && cd .. && cd ..
    fi

echo $START_PATH

set -e

if [ "x$alg" == "xCBayes" ] || [ "x$alg" == "xBinaryCBayes" ] ; then

  set -x
  echo "Preparing wikipedia data"
  rm -rf ${WORK_DIR}/wiki
  mkdir ${WORK_DIR}/wiki
  
  if [ "x$alg" == "xCBayes" ] ; then
    # use a list of 10 countries as categories
    cp $MAHOUT_HOME/examples/bin/resources/country10.txt ${WORK_DIR}/country.txt
    chmod 666 ${WORK_DIR}/country.txt
  fi
  
  if [ "x$alg" == "xBinaryCBayes" ] ; then
    # use United States and United Kingdom as categories
    cp $MAHOUT_HOME/examples/bin/resources/country2.txt ${WORK_DIR}/country.txt
    chmod 666 ${WORK_DIR}/country.txt
  fi

  if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
    echo "Copying wikipedia data to HDFS"
    set +e
    $DFSRM ${WORK_DIR}/wikixml
    $DFS -mkdir -p ${WORK_DIR}
    set -e
    $DFS -put ${WORK_DIR}/wikixml ${WORK_DIR}/wikixml
  fi

  echo "Creating sequence files from wikiXML"
  $MAHOUT_HOME/bin/mahout seqwiki -c ${WORK_DIR}/country.txt \
                                  -i ${WORK_DIR}/wikixml/enwiki-latest-pages-articles.xml \
                                  -o ${WORK_DIR}/wikipediainput
   
  # if using the 10 class problem use bigrams
  if [ "x$alg" == "xCBayes" ] ; then
    echo "Converting sequence files to vectors using bigrams"
    $MAHOUT_HOME/bin/mahout seq2sparse -i ${WORK_DIR}/wikipediainput \
                                       -o ${WORK_DIR}/wikipediaVecs \
                                       -wt tfidf \
                                       -lnorm -nv \
                                       -ow -ng 2
  fi
  
  # if using the 2 class problem try different options
  if [ "x$alg" == "xBinaryCBayes" ] ; then
    echo "Converting sequence files to vectors using unigrams and a max document frequency of 30%"
    $MAHOUT_HOME/bin/mahout seq2sparse -i ${WORK_DIR}/wikipediainput \
                                       -o ${WORK_DIR}/wikipediaVecs \
                                       -wt tfidf \
                                       -lnorm \
                                       -nv \
                                       -ow \
                                       -ng 1 \
                                       -x 30
  fi
  
  echo "Creating training and holdout set with a random 80-20 split of the generated vector dataset"
  $MAHOUT_HOME/bin/mahout split -i ${WORK_DIR}/wikipediaVecs/tfidf-vectors/ \
                                --trainingOutput ${WORK_DIR}/training \
                                --testOutput ${WORK_DIR}/testing \
                                -rp 20 \
                                -ow \
                                -seq \
                                -xm sequential

  echo "Training Naive Bayes model"
  $MAHOUT_HOME/bin/mahout trainnb -i ${WORK_DIR}/training \
                                  -o ${WORK_DIR}/model \
                                  -li ${WORK_DIR}/labelindex \
                                  -ow \
                                  -c

  echo "Self testing on training set"
  $MAHOUT_HOME/bin/mahout testnb -i ${WORK_DIR}/training \
                                 -m ${WORK_DIR}/model \
                                 -l ${WORK_DIR}/labelindex \
                                 -ow \
                                 -o ${WORK_DIR}/output \
                                 -c

  echo "Testing on holdout set: Bayes"
  $MAHOUT_HOME/bin/mahout testnb -i ${WORK_DIR}/testing \
                                 -m ${WORK_DIR}/model \
                                 -l ${WORK_DIR}/labelindex \
                                 -ow \
                                 -o ${WORK_DIR}/output \
                                 -seq

 echo "Testing on holdout set: CBayes"
  $MAHOUT_HOME/bin/mahout testnb -i ${WORK_DIR}/testing \
                                 -m ${WORK_DIR}/model -l \
                                 ${WORK_DIR}/labelindex \
                                 -ow \
                                 -o ${WORK_DIR}/output  \
                                 -c \
                                 -seq
fi

elif [ "x$alg" == "xclean" ]; then
  rm -rf $WORK_DIR
  $DFSRM $WORK_DIR
fi
# Remove the work directory
