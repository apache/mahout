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
# Requires scala, and for $HADOOP_HOME to be set.
#
# Creates test data for random forest classifier, splits data into train 
# and test sets, trains the classifier on the train set, then tests 
# model on test set.
#
# To run:  change into the mahout directory and type:
# ./examples/bin/run-rf.sh <num-rows>


if [ $# -ne 1 ]
then
  echo -e "\nThis script takes one parameter, the number of rows of random data to generate.\n"
  echo -e "Syntax: $0 <number-of-rows-of-sample-data> \n"
  exit -1
fi

WORK_DIR=/tmp/mahout-work-${USER}
INPUT="${WORK_DIR}/input"
mkdir -p $INPUT
INPUT_PATH="${INPUT}/rf-input.csv"

# Set commands for dfs
source ./examples/bin/set-dfs-commands.sh

# Create test data
numrows=$1
echo "Writing random data to $INPUT_PATH"
./examples/bin/create-rf-data.sh $numrows $INPUT_PATH

# Put the test file in HDFS
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
  echo "Copying random data to HDFS"
  set +e
  $DFSRM $WORK_DIR
  $DFS -mkdir -p $INPUT
  set -e
  $DFS -put $INPUT_PATH $INPUT
fi

# Split original file into train and test
echo "Creating training and holdout set with a random 60-40 split of the generated vector dataset"
./bin/mahout split \
  -i $INPUT \
  --trainingOutput ${WORK_DIR}/train.csv \
  --testOutput ${WORK_DIR}/test.csv \
  --randomSelectionPct 40 --overwrite -xm sequential

# Describe input file schema
# Note:  "-d 4 N L" indicates four numerical fields and one label, as built by the step above.
./bin/mahout describe -p $INPUT_PATH -f ${WORK_DIR}/info -d 4 N L

# Train rf model
echo
echo "Training random forest."
echo
./bin/mahout buildforest -DXmx10000m -Dmapred.max.split.size=1000000 -d ${WORK_DIR}/train.csv -ds ${WORK_DIR}/info -sl 7 -p -t 500 -o ${WORK_DIR}/forest

# Test predictions
echo
echo "Testing predictions on test set."
echo
./bin/mahout testforest -DXmx10000m -Dmapred.output.compress=false -i ${WORK_DIR}/test.csv -ds ${WORK_DIR}/info -m ${WORK_DIR}/forest -a -mr -o ${WORK_DIR}/predictions

# Remove old files
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ]
then
  $DFSRM $WORK_DIR
fi
rm -r $WORK_DIR

