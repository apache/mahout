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

# Instructions:
#
# This script automatically downloads yalefaces example and applies eigenfaces
# extraction and classification of test data.
#
# Based on https://chimpler.wordpress.com/2013/04/17/generating-eigenfaces-with-mahout-svd-to-recognize-person-faces/
#
# To run:  change into the mahout directory and type:
#  export MAHOUT_LOCAL=true
set -e

if [ "$1" = "--help" ] || [ "$1" = "--?" ]; then
  echo "This script automatically downloads yalefaces example, splits the dataset and applies eigenfaces extraction and classification of test data."
  exit
fi

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then 
  cd $SCRIPT_PATH
fi
START_PATH=`pwd`

# Set commands for dfs
source ${START_PATH}/set-dfs-commands.sh

MAHOUT="../../bin/mahout"

if [ ! -e "$MAHOUT" ]; then
  echo "Can't find mahout driver in $MAHOUT, cwd `pwd`, exiting.."
  exit 1
fi

if [[ -z "$MAHOUT_WORK_DIR" ]]; then
  WORK_DIR="/tmp/mahout-work-${USER}"
else
  WORK_DIR="$MAHOUT_WORK_DIR"
fi

export MAHOUT_LOCAL=true

actions=( sad normal happy noglasses clean)
if [ -n "$1" ]; then
  choice=$1
else
  echo "Please select a number to choose the corresponding action:"
  for i in $(seq 1 $(( ${#actions[@]} - 1))); do
    echo "$i. ${actions[$i - 1]} -- compute eigenfaces of Yale dataset and use sad faces for testing" 
  done
  echo "${#actions[@]}. ${actions[${#actions[@]} - 1]} -- cleans up the work area in $WORK_DIR"
  read -p "Enter your choice : " choice
fi

echo "ok. You chose $choice and we'll use ${actions[$choice-1]}"
action=${actions[$choice-1]}

if [ "x$action" == "xclean" ]; then
  rm -rf $WORK_DIR || $DFSRM $WORK_DIR || true
  exit 1
else
  $DFS -mkdir -p $WORK_DIR
  mkdir -p $WORK_DIR
  echo "Creating work directory at ${WORK_DIR}"
fi

if [ ! -e "${WORK_DIR}/yalefaces-cleaned" ]; then
  # the yale faces should be normalized
  if [ ! -e "${WORK_DIR}/yalefaces-raw" ]; then
    # For this example we will use the Yale Face Database http://vision.ucsd.edu/content/yalefaces-face-database
    # More datasets can be found in http://www.face-rec.org/databases/
    # http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip
    mkdir "${WORK_DIR}/yalefaces-raw"
    wget -c http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip -O "${WORK_DIR}/yalefaces-raw/yalefaces.zip"
    #cp ../../../work/yalefaces.zip "${WORK_DIR}/yalefaces-raw/yalefaces.zip"
    unzip "${WORK_DIR}/yalefaces-raw/yalefaces.zip" -d "${WORK_DIR}/yalefaces-raw/"
  fi
  mkdir -p "${WORK_DIR}/yalefaces-cleaned"
  # do not move the .gif files - they have to be handled differently
  cp "${WORK_DIR}/yalefaces-raw/yalefaces/"subject*[^f] "${WORK_DIR}/yalefaces-cleaned"
  cp "${WORK_DIR}/yalefaces-raw/yalefaces/subject01.gif" "${WORK_DIR}/yalefaces-cleaned/subject01.centerlight"
  # there is subject01.glasses.gif which is duplicate of subject01.glasses 
fi

TRAIN_DIR="${WORK_DIR}/yalefaces-train/"
TEST_DIR="${WORK_DIR}/yalefaces-test"
if [ ! -e "${TEST_DIR}/subject01.$action" ]; then
  echo "Splitting dataset into training and testing (*.$action for testing)"
  # the requested action is not found in the train/test set - start all over
  rm -rf "${TRAIN_DIR}" || true
  mkdir -p "${TRAIN_DIR}"
  cp "${WORK_DIR}/yalefaces-cleaned/"* "${TRAIN_DIR}"
  mkdir -p "${TEST_DIR}"
  rm -f "${TEST_DIR}"/subject* || true
  mv "${TRAIN_DIR}"/subject*.$action  ${TEST_DIR}
  # download additional images, note that these should be the same size
  echo "Adding examples not in Yale dataset for testing"
  for sample in brucelee cat hamburger; do
    wget https://raw.githubusercontent.com/fredang/mahout-eigenface-example/master/images/yalefaces-test/$sample.gif -O "${TEST_DIR}/$sample.gif"
  done
fi

if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
  echo "Copying eigenfaces data to Hadoop"
  set +e
  $DFSRM "${TRAIN_DIR}"
  $DFSRM "${TEST_DIR}"
  $DFS -mkdir -p "${TRAIN_DIR}/"
  $DFS -mkdir -p "${TEST_DIR}/"
  $DFS -put "${TRAIN_DIR}" "${TRAIN_DIR}"
  $DFS -put "${TEST_DIR}" "${TEST_DIR}"
  set -e
fi

width=$(identify -format "%w" $(ls -1d "${TRAIN_DIR}/"* | head -n 1))  # 80
height=$(identify -format "%h" $(ls -1d "${TRAIN_DIR}/"* | head -n 1))  # 60

echo "Generating covariance matrix from test set"
set +e
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then 
  $DFS -mkdir -p "${WORK_DIR}/eigenfaces/covariance"
else
  mkdir -p "${WORK_DIR}/eigenfaces/covariance"
fi
set -e
"$MAHOUT" org.apache.mahout.eigenfaces.GenerateCovarianceMatrix "$width" "$height" "${TRAIN_DIR}" "${WORK_DIR}/eigenfaces/covariance"

if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
  $DFS -mv "${WORK_DIR}/eigenfaces/covariance/diffmatrix.seq" "${WORK_DIR}/eigenfaces/"
  $DFS -mv "${WORK_DIR}/eigenfaces/covariance/mean-image.gif" "${WORK_DIR}/eigenfaces/"
else
  mv "${WORK_DIR}/eigenfaces/covariance/diffmatrix.seq" "${WORK_DIR}/eigenfaces/"
  mv "${WORK_DIR}/eigenfaces/covariance/mean-image.gif" "${WORK_DIR}/eigenfaces/"
fi

echo "Running SVD"
samples=$(ls -1 "${TRAIN_DIR}" | wc -l)
rank=$(( $samples / 3 ))
"$MAHOUT" svd --input "${WORK_DIR}/eigenfaces/covariance/covariance.seq" --numRows $samples --numCols $samples --rank $rank --output "${WORK_DIR}/eigenfaces/svd/"

echo "Cleaning SVD"
# from rawEigenvectors create cleanEigenvectors
"$MAHOUT" cleansvd -ci "${WORK_DIR}/eigenfaces/covariance/covariance.seq" -ei "${WORK_DIR}/eigenfaces/svd/" -o "${WORK_DIR}/eigenfaces/clean-svd/"

echo "Compute eigenfaces on training set"
"$MAHOUT" org.apache.mahout.eigenfaces.ComputeEigenFaces "${WORK_DIR}/eigenfaces/clean-svd/cleanEigenvectors" "${WORK_DIR}/eigenfaces/diffmatrix.seq" "${WORK_DIR}/eigenfaces/mean-image.gif" $width $height ${TRAIN_DIR} "${WORK_DIR}/eigenfaces/eigenfaces"

echo "Compute similarity for test set"
"$MAHOUT" org.apache.mahout.eigenfaces.ComputeDistance "${WORK_DIR}/eigenfaces/eigenfaces/eigenfaces.seq" "${WORK_DIR}/eigenfaces/mean-image.gif" "${WORK_DIR}/eigenfaces/eigenfaces/weights.seq" $width $height "${TRAIN_DIR}" "${TEST_DIR}" "${WORK_DIR}/eigenfaces/eigenfaces"
