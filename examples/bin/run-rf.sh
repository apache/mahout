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
WORK_DIR=/tmp/mahout-work-${USER}/
input="rf-input.csv"


# Remove old files
echo
echo "Removing old temp files if they exist; this will mention they're not there if not."
echo
$HADOOP_HOME/bin/hadoop fs -rmr -skipTrash $WORK_DIR forest
$HADOOP_HOME/bin/hadoop fs -mkdir $WORK_DIR

# Create test data
numrows=$1
echo
echo "Writing random data to $input"
./examples/bin/create-rf-data.sh $numrows $input

# Put the test file in HDFS
$HADOOP_HOME/bin/hadoop fs -rmr -skipTrash ${WORK_DIR}
$HADOOP_HOME/bin/hadoop fs -mkdir -p ${WORK_DIR}/input
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
  HADOOP="$HADOOP_HOME/bin/hadoop"
  if [ ! -e $HADOOP ]; then
    echo "Can't find hadoop in $HADOOP, exiting"
    exit 1
  fi
fi
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
  echo "Copying random data to HDFS"
  set +e
  $HADOOP dfs -rmr ${WORK_DIR}
  set -e
  $HADOOP dfs -put $input ${WORK_DIR}/input/$input
fi

# Split original file into train and test
echo "Creating training and holdout set with a random 60-40 split of the generated vector dataset"
./bin/mahout split \
  -i ${WORK_DIR}/input \
  --trainingOutput ${WORK_DIR}/train.csv \
  --testOutput ${WORK_DIR}/test.csv \
  --randomSelectionPct 40 --overwrite -xm sequential

# Describe input file schema
# Note:  "-d 4 N L" indicates four numerical fields and one label, as built by the step above.
./bin/mahout describe -p $WORK_DIR/input/$input -f $WORK_DIR/info -d 4 N L

# Train rf model
echo
echo "Training random forest."
echo
./bin/mahout buildforest -DXmx10000m -Dmapred.max.split.size=1000000 -d $WORK_DIR/train.csv -ds $WORK_DIR/info -sl 7 -p -t 500 -o $WORK_DIR/forest

# Test predictions
echo
echo "Testing predictions on test set."
echo
./bin/mahout testforest -DXmx10000m -Dmapred.output.compress=false -i $WORK_DIR/test.csv -ds $WORK_DIR/info -m $WORK_DIR/forest -a -mr -o $WORK_DIR/predictions

# Remove old files
$HADOOP_HOME/bin/hadoop fs -rmr -skipTrash $WORK_DIR
rm $input

