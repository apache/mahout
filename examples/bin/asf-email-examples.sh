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
# You will need to download or otherwise obtain some or all of the Amazon ASF Email Public Dataset (http://aws.amazon.com/datasets/7791434387204566) to use this script.
# To obtain a full copy you will need to launch an EC2 instance and mount the dataset to download it, otherwise you can get a sample of it at
# http://www.lucidimagination.com/devzone/technical-articles/scaling-mahout
# Also, see http://www.ibm.com/developerworks/java/library/j-mahout-scaling/ for more info

function fileExists() {
  if ([ "$MAHOUT_LOCAL" != "" ] && [ ! -e "$1" ]) || ([ "$MAHOUT_LOCAL" == "" ] && ! hadoop fs -test -e /user/$USER/$1); then
    return 1 # file doesn't exist
  else
    return 0 # file exists
  fi
}

function removeFolder() {
  if [ "$MAHOUT_LOCAL" == "" ]; then
    rm -rf $1
  else
    if fileExists "$1"; then
      hadoop fs -rmr /user/$USER/$1
    fi
  fi	
}

if [ "$1" = "--help" ] || [ "$1" = "--?" ]; then
  echo "This script runs recommendation, classification and clustering of the ASF Email Public Dataset, as hosted on Amazon (http://aws.amazon.com/datasets/7791434387204566).  Requires download."
  exit
fi

if [ -z "$2" ]; then
  echo "Usage: asf-email-examples.sh input_path output_path"
  exit
fi

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then
  cd $SCRIPT_PATH
fi
START_PATH=`pwd`
MAHOUT="../../bin/mahout"
ASF_ARCHIVES=$1
OUT=$2

algorithm=( recommender clustering classification clean )
if [ -n "$3" ]; then
  choice=$3
else
  echo "Please select a number to choose the corresponding algorithm to run"
  echo "1. ${algorithm[0]}"
  echo "2. ${algorithm[1]}"
  echo "3. ${algorithm[2]}"
  echo "4. ${algorithm[3]} -- cleans up the work area -- all files under the work area will be deleted"
  read -p "Enter your choice : " choice
fi
echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
alg=${algorithm[$choice-1]}

if [ "x$alg" == "xrecommender" ]; then
  # convert the mail to seq files
  MAIL_OUT="$OUT/prefs/seq-files"
  if ! fileExists "$MAIL_OUT/chunk-0"; then
    echo "Converting Mail files to Sequence Files"
    $MAHOUT org.apache.mahout.text.SequenceFilesFromMailArchives --charset "UTF-8" --from --references --input $ASF_ARCHIVES --output $MAIL_OUT --separator " ::: "
  fi
  PREFS="$OUT/prefs/input"
  PREFS_TMP="$OUT/prefs/tmp"
  PREFS_REC_INPUT="$OUT/prefs/input/recInput"
  RECS_OUT=$"$OUT/prefs/recommendations"
  # prep for recs
  if ! fileExists "$PREFS/fromIds-dictionary-0"; then
    echo "Prepping Sequence files for Recommender"
    $MAHOUT org.apache.mahout.cf.taste.example.email.MailToPrefsDriver --input $MAIL_OUT --output $PREFS --overwrite --separator " ::: "
  fi
  removeFolder "$PREFS_TMP"
  removeFolder "$RECS_OUT"
  # run the recs
  echo "Run the recommender"
  $MAHOUT recommenditembased --input $PREFS_REC_INPUT --output $RECS_OUT --tempDir $PREFS_TMP --similarityClassname SIMILARITY_LOGLIKELIHOOD

#clustering
elif [ "x$alg" == "xclustering" ]; then
  MAIL_OUT="$OUT/clustering/seq-files"
  SEQ2SP="$OUT/clustering/seq2sparse"
  algorithm=( kmeans dirichlet minhash )

  if [ -n "$4" ]; then
    choice=$4
  else
    echo "Please select a number to choose the corresponding algorithm to run"
    echo "1. ${algorithm[0]}"
    echo "2. ${algorithm[1]}"
    echo "3. ${algorithm[2]}"
    read -p "Enter your choice : " choice
  fi

  echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
  nbalg=${algorithm[$choice-1]}
  if [ "x$nbalg" == "xkmeans"  ] || [ "x$nbalg" == "xdirichlet" ]; then
    if [ -n "$5" ]; then
      numClusters=$5
    else
      echo "How many clusters would you like to generate:"
      read -p "Enter your choice : " numClusters
    fi
  fi
  if ! fileExists "$MAIL_OUT/chunk-0"; then
    echo "Converting Mail files to Sequence Files"
    $MAHOUT org.apache.mahout.text.SequenceFilesFromMailArchives --charset "UTF-8" --subject --body --input $ASF_ARCHIVES --output $MAIL_OUT
  fi

  #convert to sparse vectors -- use the 2 norm (Euclidean distance) and lop of some of the common terms

  if ! fileExists "$SEQ2SP/dictionary.file-0"; then
    echo "Converting the files to sparse vectors"
    $MAHOUT seq2sparse --input $MAIL_OUT --output $SEQ2SP --norm 2 --weight TFIDF --namedVector --maxDFPercent 90 --minSupport 2 --analyzerName org.apache.mahout.text.MailArchivesClusteringAnalyzer
  fi
  if [ "x$nbalg" == "xkmeans" ]; then
    CLUST_OUT="$OUT/clustering/kmeans"
    echo "Running K-Means with K = $numClusters"
    $MAHOUT kmeans --input "$SEQ2SP/tfidf-vectors" --output $CLUST_OUT -k $numClusters --maxIter 20 --distanceMeasure org.apache.mahout.common.distance.CosineDistanceMeasure --clustering --method mapreduce --clusters "$CLUST_OUT/clusters"
  elif [ "x$nbalg" == "xdirichlet"  ]; then
    CLUST_OUT="$OUT/clustering/dirichlet"
    echo "Running Dirichlet with K = $numClusters"
    $MAHOUT dirichlet --input "$SEQ2SP/tfidf-vectors" --output $CLUST_OUT -k $numClusters --maxIter 20 --distanceMeasure org.apache.mahout.common.distance.CosineDistanceMeasure --method mapreduce
  elif [ "x$nbalg" == "xminhash"  ]; then
    CLUST_OUT="$OUT/clustering/minhash"
    echo "Running Minhash"
    $MAHOUT minhash --input "$SEQ2SP/tfidf-vectors" --output $CLUST_OUT
  fi

#classification
elif [ "x$alg" == "xclassification" ]; then
  algorithm=( standard complementary sgd )
  echo ""
  if [ -n "$4" ]; then
    choice=$4
  else
    echo "Please select a number to choose the corresponding algorithm to run"
    echo "1. ${algorithm[0]}"
    echo "2. ${algorithm[1]}"
    echo "3. ${algorithm[2]}"
    read -p "Enter your choice : " choice
  fi
  
  echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
  classAlg=${algorithm[$choice-1]}

  if [ "x$classAlg" == "xsgd"  ]; then
    if [ -n "$5" ]; then
      numLabels=$5
    else
      echo "How many labels/projects are there in the data set:"
      read -p "Enter your choice : " numLabels
    fi
  fi
  #Convert mail to be formatted as:
  # label\ttext
  # One per line
  # the label is the project_name_mailing_list, as in tomcat.apache.org_dev
  #Convert to vectors
  if [ "x$classAlg" == "xstandard" ] || [ "x$classAlg" == "xcomplementary" ]; then
	set -x
    CLASS="$OUT/classification/bayesian"
    MAIL_OUT="$CLASS/seq-files"
    SEQ2SP="$CLASS/seq2sparse"
    SPLIT="$CLASS/splits"
    TRAIN="$SPLIT/train"
    TEST="$SPLIT/test"
    TEST_OUT="$CLASS/test-results"
    LABEL="$SPLIT/labels"
    if ! fileExists "$MAIL_OUT/chunk-0"; then
      echo "Converting Mail files to Sequence Files"
      $MAHOUT org.apache.mahout.text.SequenceFilesFromMailArchives --charset "UTF-8" --subject --body --input $ASF_ARCHIVES --output $MAIL_OUT -chunk 768 --stripQuoted
    fi
    if ! fileExists "$SEQ2SP/dictionary.file-0"; then
      echo "Converting the files to sparse vectors"
      # $MAHOUT seq2sparse --input $MAIL_OUT --output $SEQ2SP --norm 2 --weight TFIDF --namedVector -lnorm --maxDFPercent 90 --minSupport 2 --analyzerName org.apache.mahout.text.MailArchivesClusteringAnalyzer -chunk 1000
      $MAHOUT seq2encoded --input $MAIL_OUT --output $SEQ2SP --analyzerName org.apache.mahout.text.MailArchivesClusteringAnalyzer --cardinality 100000 -ow
	fi
    if ! fileExists "$TRAIN/part-m-00000"; then
      #setup train/test files
      echo "Creating training and test inputs"
      $MAHOUT split --input $SEQ2SP --trainingOutput $TRAIN --testOutput $TEST --randomSelectionPct 20 --overwrite --sequenceFiles -xm sequential
    fi
    MODEL="$CLASS/model"
    if [ "x$classAlg" == "xstandard" ]; then
      echo "Running Standard Training"
      $MAHOUT trainnb -i $TRAIN -o $MODEL --extractLabels --labelIndex $LABEL --overwrite
      echo "Running Test"
      $MAHOUT testnb -i $TEST -o $TEST_OUT -m $MODEL --labelIndex $LABEL --overwrite

    elif [ "x$classAlg" == "xcomplementary"  ]; then
      echo "Running Complementary Training"
      $MAHOUT trainnb -i $TRAIN -o $MODEL --extractLabels --labelIndex $LABEL --overwrite --trainComplementary
      echo "Running Complementary Test"
      $MAHOUT testnb -i $TEST -o $TEST_OUT -m $MODEL --labelIndex $LABEL --overwrite --testComplementary
    fi
  elif [ "x$classAlg" == "xsgd"  ]; then
    CLASS="$OUT/classification/sgd"
    MAIL_OUT="$CLASS/seq-files"
    SEQ2SP="$CLASS/seq2encoded"
    SEQ2SPLABEL="$CLASS/labeled"
    SPLIT="$CLASS/splits"
    TRAIN="$SPLIT/train"
    TEST="$SPLIT/test"
    MAPREDOUT="$SPLIT/mapRedOut"
    TEST_OUT="$CLASS/test-results"
    MODELS="$CLASS/models"
    LABEL="$SPLIT/labels"
    if ! fileExists "$MAIL_OUT/chunk-0"; then
      echo "Converting Mail files to Sequence Files"
      $MAHOUT org.apache.mahout.text.SequenceFilesFromMailArchives --charset "UTF-8" --subject --body --input $ASF_ARCHIVES --output $MAIL_OUT --stripQuoted -chunk 768
    fi
    echo "Converting the files to sparse vectors in $SEQ2SP"
    if ! fileExists "$SEQ2SP/part-m-00000"; then
      $MAHOUT seq2encoded --input $MAIL_OUT --output $SEQ2SP --analyzerName org.apache.mahout.text.MailArchivesClusteringAnalyzer --cardinality 100000
    fi
    #We need to modify the vectors to have a better label
    echo "Converting vector labels"
    $MAHOUT org.apache.mahout.classifier.email.PrepEmailVectorsDriver --input "$SEQ2SP" --output $SEQ2SPLABEL --overwrite
    if ! fileExists "$MAPREDOUT/training-r-00000"; then
      #setup train/test files
      echo "Creating training and test inputs from $SEQ2SPLABEL"
      $MAHOUT split --input $SEQ2SPLABEL --mapRedOutputDir $MAPREDOUT  --randomSelectionPct 20 --overwrite --sequenceFiles --method mapreduce
    fi
    MODEL="$MODELS/asf.model"


    echo "Running SGD Training"
    $MAHOUT org.apache.mahout.classifier.sgd.TrainASFEmail -i $MAPREDOUT/ -o $MODELS --categories $numLabels --cardinality 100000
    echo "Running Test"
    $MAHOUT org.apache.mahout.classifier.sgd.TestASFEmail --input $MAPREDOUT/ --model $MODEL

  fi
elif [ "x$alg" == "xclean" ]; then
  echo "Are you sure you really want to remove all files under $OUT:"
  read -p "Enter your choice (y/n): " answer
  if [ "x$answer" == "xy" ] || [ "x$answer" == "xY" ]; then
    echo "Cleaning out $OUT";
	removeFolder "$OUT"
  fi
fi


