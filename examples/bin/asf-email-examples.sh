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

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then
  cd $SCRIPT_PATH
fi
START_PATH=`pwd`
MAHOUT="../../bin/mahout"
ASF_ARCHIVES=$1
OUT=$2
OVER=$3
export MAHOUT_HEAPSIZE=2048

if [ "$1" = "-ni" ]; then
  alg=rec
else
  algorithm=( recommender clustering classification )

  echo "Please select a number to choose the corresponding algorithm to run"
  echo "1. ${algorithm[0]}"
  echo "2. ${algorithm[1]}"
  echo "3. ${algorithm[2]}"
  read -p "Enter your choice : " choice

  echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
  alg=${algorithm[$choice-1]}
fi


if [ "x$alg" == "xrecommender" ]; then
  # convert the mail to seq files
  MAIL_OUT="$OUT/prefs/seq-files"
  if [ "x$OVER" == "xover" ] || [ ! -e "$MAIL_OUT/chunk-0" ]; then
    echo "Converting Mail files to Sequence Files"
    $MAHOUT org.apache.mahout.text.SequenceFilesFromMailArchives --charset "UTF-8" --from --references --input $ASF_ARCHIVES --output $MAIL_OUT --separator " ::: "
  fi
  PREFS="$OUT/prefs/input"
  PREFS_TMP="$OUT/prefs/tmp"
  PREFS_REC_INPUT="$OUT/prefs/input/recInput"
  RECS_OUT=$"$OUT/prefs/recommendations"
  # prep for recs
  if [ "x$OVER" == "xover" ] || [ ! -e "$PREFS/fromIds-dictionary-0" ]; then
    echo "Prepping Sequence files for Recommender"
    $MAHOUT org.apache.mahout.cf.taste.example.email.MailToPrefsDriver --input $MAIL_OUT --output $PREFS --overwrite --separator " ::: "
  fi
  # run the recs
  echo "Run the recommender"
  $MAHOUT recommenditembased --input $PREFS_REC_INPUT --output $RECS_OUT --tempDir $PREFS_TMP --similarityClassname SIMILARITY_LOGLIKELIHOOD

#clustering
elif [ "x$alg" == "xclustering" ]; then
  MAIL_OUT="$OUT/clustering/seq-files"
  SEQ2SP="$OUT/clustering/seq2sparse"
  algorithm=( kmeans dirichlet minhash )

  echo "Please select a number to choose the corresponding algorithm to run"
  echo "1. ${algorithm[0]}"
  echo "2. ${algorithm[1]}"
  echo "3. ${algorithm[2]}"
  read -p "Enter your choice : " choice

  echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
  nbalg=${algorithm[$choice-1]}
  if [ "x$OVER" == "xover" ] || [ ! -e "$MAIL_OUT/chunk-0" ]; then
    echo "Converting Mail files to Sequence Files"
    $MAHOUT org.apache.mahout.text.SequenceFilesFromMailArchives --charset "UTF-8" --subject --body --input $ASF_ARCHIVES --output $MAIL_OUT
  fi

  #convert to sparse vectors -- use the 2 norm (Euclidean distance) and lop of some of the common terms

  if [ "x$OVER" == "xover" ] || [ ! -e "$SEQ2SP/dictionary.file-0" ]; then
    echo "Converting the files to sparse vectors"
    $MAHOUT seq2sparse --input $MAIL_OUT --output $SEQ2SP --norm 2 --weight TFIDF --namedVector --maxDFPercent 90 --minSupport 2 --analyzerName org.apache.mahout.text.MailArchivesClusteringAnalyzer
  fi
  if [ "x$nbalg" == "xkmeans" ]; then
    CLUST_OUT="$OUT/clustering/kmeans"
    echo "Running K-Means"
    $MAHOUT kmeans --input "$SEQ2SP/tfidf-vectors" --output $CLUST_OUT -k 50 --maxIter 20 --distanceMeasure org.apache.mahout.common.distance.CosineDistanceMeasure --clustering --method mapreduce --clusters "$CLUST_OUT/clusters"
  elif [ "x$nbalg" == "xdirichlet"  ]; then
    CLUST_OUT="$OUT/clustering/dirichlet"
    echo "Running Dirichlet"
    $MAHOUT dirichlet --input "$SEQ2SP/tfidf-vectors" --output $CLUST_OUT -k 50 --maxIter 20 --distanceMeasure org.apache.mahout.common.distance.CosineDistanceMeasure --method mapreduce
  elif [ "x$nbalg" == "xminhash"  ]; then
    CLUST_OUT="$OUT/clustering/minhash"
    echo "Running Minhash"
    $MAHOUT minhash --input "$SEQ2SP/tfidf-vectors" --output $CLUST_OUT
  fi

#classification
elif [ "x$alg" == "xclassification" ]; then
  algorithm=( standard complementary sgd )

  echo "Please select a number to choose the corresponding algorithm to run"
  echo "1. ${algorithm[0]}"
  echo "2. ${algorithm[1]}"
  echo "3. ${algorithm[2]}"
  read -p "Enter your choice : " choice

  echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]}"
  classAlg=${algorithm[$choice-1]}

  if [ "x$classAlg" == "xsgd"  ]; then
    echo "How many labels/projects are there in the data set:"
    read -p "Enter your choice : " numLabels
  fi
  #Convert mail to be formatted as:
  # label\ttext
  # One per line
  # the label is the project_name_mailing_list, as in tomcat.apache.org_dev
  #Convert to vectors
  if [ "x$classAlg" == "xstandard" ] || [ "x$classAlg" == "xcomplementary" ]; then
    CLASS="$OUT/classification/bayesian"
    MAIL_OUT="$CLASS/seq-files"
    SEQ2SP="$CLASS/seq2sparse"
    SEQ2SPLABEL="$CLASS/labeled"
    SPLIT="$CLASS/splits"
    TRAIN="$SPLIT/train"
    TEST="$SPLIT/test"
    TEST_OUT="$CLASS/test-results"
    LABEL="$SPLIT/labels"
    if [ "x$OVER" == "xover" ] || [ ! -e "$MAIL_OUT/chunk-0" ]; then
      echo "Converting Mail files to Sequence Files"
      $MAHOUT org.apache.mahout.text.SequenceFilesFromMailArchives --charset "UTF-8" --subject --body --input $ASF_ARCHIVES --output $MAIL_OUT
    fi
    if [ "x$OVER" == "xover" ] || [ ! -e "$SEQ2SP/dictionary.file-0" ]; then
      echo "Converting the files to sparse vectors"
      $MAHOUT seq2sparse --input $MAIL_OUT --output $SEQ2SP --norm 2 --weight TFIDF --namedVector --maxDFPercent 90 --minSupport 2 --analyzerName org.apache.mahout.text.MailArchivesClusteringAnalyzer
      #We need to modify the vectors to have a better label
      echo "Converting vector labels"
      $MAHOUT org.apache.mahout.classifier.email.PrepEmailVectorsDriver --input "$SEQ2SP/tfidf-vectors" --output $SEQ2SPLABEL --overwrite --maxItemsPerLabel 1000
    fi
    if [ "x$OVER" == "xover" ] || [ ! -e "$TRAIN/part-m-00000" ]; then
      #setup train/test files
      echo "Creating training and test inputs"
      $MAHOUT split --input $SEQ2SPLABEL --trainingOutput $TRAIN --testOutput $TEST --randomSelectionPct 20 --overwrite --sequenceFiles
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
    TEST_OUT="$CLASS/test-results"
    MODELS="$CLASS/models"
    LABEL="$SPLIT/labels"
    if [ "x$OVER" == "xover" ] || [ ! -e "$MAIL_OUT/chunk-0" ]; then
      echo "Converting Mail files to Sequence Files"
      $MAHOUT org.apache.mahout.text.SequenceFilesFromMailArchives --charset "UTF-8" --subject --body --input $ASF_ARCHIVES --output $MAIL_OUT
    fi
    echo "Converting the files to sparse vectors in $SEQ2SP"
    $MAHOUT seq2encoded --input $MAIL_OUT --output $SEQ2SP --analyzerName org.apache.mahout.text.MailArchivesClusteringAnalyzer
    #We need to modify the vectors to have a better label
    echo "Converting vector labels"
    $MAHOUT org.apache.mahout.classifier.email.PrepEmailVectorsDriver --input "$SEQ2SP" --output $SEQ2SPLABEL --overwrite
    if [ "x$OVER" == "xover" ] || [ ! -e "$TRAIN/part-m-00000" ]; then
      #setup train/test files
      echo "Creating training and test inputs from $SEQ2SPLABEL"
      $MAHOUT split --input $SEQ2SPLABEL --trainingOutput $TRAIN --testOutput $TEST --randomSelectionPct 20 --overwrite --sequenceFiles
    fi
    MODEL="$MODELS/asf.model"


    echo "Running SGD Training"
    $MAHOUT org.apache.mahout.classifier.sgd.TrainASFEmail $TRAIN $MODELS $numLabels 5000
    echo "Running Test"
    $MODEL="$MODELS/asf.model"
    $MAHOUT org.apache.mahout.classifier.sgd.TestASFEmail --input $TEST --model $MODEL

  fi
fi


