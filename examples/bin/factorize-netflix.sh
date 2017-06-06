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
# You can only use this script in conjunction with the Netflix dataset. Unpack the Netflix dataset and provide the
# following:
#
#   1) the path to the folder 'training_set' that contains all the movie rating files
#   2) the path to the file 'qualifying.txt' that contains the user,item pairs to predict
#   3) the path to the file 'judging.txt' that contains the ratings of user,item pairs to predict for
#
# To run:
#  ./factorize-netflix.sh /path/to/training_set/ /path/to/qualifying.txt /path/to/judging.txt

echo "Note this script has been deprecated due to the lack of access to the Netflix data set."
exit 1

if [ "$1" = "--help" ] || [ "$1" = "--?" ]; then
  echo "This script runs the ALS Recommender on the Netflix data set."
  echo "Syntax: $0 /path/to/training_set/ /path/to/qualifying.txt /path/to/judging.txt\n"
  exit
fi

if [ $# -ne 3 ]
then
  echo -e "Syntax: $0 /path/to/training_set/ /path/to/qualifying.txt /path/to/judging.txt\n"
  exit -1
fi

MAHOUT="../../bin/mahout"

if [[ -z "$MAHOUT_WORK_DIR" ]]; then
  WORK_DIR=/tmp/mahout-work-${USER}
else
  WORK_DIR=$MAHOUT_WORK_DIR
fi

START_PATH=`pwd`

# Set commands for dfs
source ${START_PATH}/set-dfs-commands.sh

echo "Preparing data..."
$MAHOUT org.apache.mahout.cf.taste.hadoop.example.als.netflix.NetflixDatasetConverter $1 $2 $3 ${WORK_DIR}

# run distributed ALS-WR to factorize the rating matrix defined by the training set
$MAHOUT parallelALS --input ${WORK_DIR}/trainingSet/ratings.tsv --output ${WORK_DIR}/als/out \
    --tempDir ${WORK_DIR}/als/tmp --numFeatures 25 --numIterations 10 --lambda 0.065 --numThreadsPerSolver 4

# compute predictions against the probe set, measure the error
$MAHOUT evaluateFactorization --input ${WORK_DIR}/probeSet/ratings.tsv --output ${WORK_DIR}/als/rmse/ \
    --userFeatures ${WORK_DIR}/als/out/U/ --itemFeatures ${WORK_DIR}/als/out/M/ --tempDir ${WORK_DIR}/als/tmp

if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then

  # print the error, should be around 0.923
  echo -e "\nRMSE is:\n"
  $DFS -tail ${WORK_DIR}/als/rmse/rmse.txt
  echo -e "\n"
  echo "removing work directory"
  set +e
  $DFSRM ${WORK_DIR}

else

  # print the error, should be around 0.923
  echo -e "\nRMSE is:\n"
  cat ${WORK_DIR}/als/rmse/rmse.txt
  echo -e "\n"
  echo "removing work directory"
  rm -rf ${WORK_DIR}

fi

