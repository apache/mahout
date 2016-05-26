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
# Downloads the Synthetic control dataset and prepares it for clustering
#
# To run:  change into the mahout directory and type:
#  examples/bin/cluster-syntheticcontrol.sh

if [ "$1" = "--help" ] || [ "$1" = "--?" ]; then
  echo "This script clusters the Synthetic Control data set.  The data set is downloaded automatically."
  exit
fi

algorithm=( kmeans fuzzykmeans )
if [ -n "$1" ]; then
  choice=$1
else
  echo "Please select a number to choose the corresponding clustering algorithm"
  echo "1. ${algorithm[0]} clustering"
  echo "2. ${algorithm[1]} clustering"
  read -p "Enter your choice : " choice
fi
echo "ok. You chose $choice and we'll use ${algorithm[$choice-1]} Clustering"
clustertype=${algorithm[$choice-1]}

SCRIPT_PATH=${0%/*}
if [ "$0" != "$SCRIPT_PATH" ] && [ "$SCRIPT_PATH" != "" ]; then
  cd $SCRIPT_PATH
fi
START_PATH=`pwd`

# Set commands for dfs
source ${START_PATH}/set-dfs-commands.sh

if [[ -z "$MAHOUT_WORK_DIR" ]]; then
  WORK_DIR=/tmp/mahout-work-${USER}
else
  WORK_DIR=$MAHOUT_WORK_DIR
fi

echo "creating work directory at ${WORK_DIR}"
mkdir -p ${WORK_DIR}
if [ ! -f ${WORK_DIR}/synthetic_control.data ]; then
  if [ -n "$2" ]; then
    cp $2 ${WORK_DIR}/.
  else
    echo "Downloading Synthetic control data"
    curl http://archive.ics.uci.edu/ml/databases/synthetic_control/synthetic_control.data  -o ${WORK_DIR}/synthetic_control.data
  fi
fi
if [ ! -f ${WORK_DIR}/synthetic_control.data ]; then
  echo "Couldn't download synthetic control"
  exit 1
fi
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ]; then
  echo "Checking the health of DFS..."
  $DFS -ls /
  if [ $? -eq 0 ];then 
    echo "DFS is healthy... "
    echo "Uploading Synthetic control data to HDFS"
    $DFSRM ${WORK_DIR}/testdata
    $DFS -mkdir -p ${WORK_DIR}/testdata
    $DFS -put ${WORK_DIR}/synthetic_control.data ${WORK_DIR}/testdata
    echo "Successfully Uploaded Synthetic control data to HDFS "

    options="--input ${WORK_DIR}/testdata --output ${WORK_DIR}/output --maxIter 10 --convergenceDelta 0.5"

    if [ "${clustertype}" == "kmeans" ]; then
      options="${options} --numClusters 6"
      # t1 & t2 not used if --numClusters specified, but parser requires input
      options="${options} --t1 1 --t2 2"
      ../../bin/mahout org.apache.mahout.clustering.syntheticcontrol."${clustertype}".Job ${options}
    else
      options="${options} --m 2.0f --t1 80 --t2 55"
      ../../bin/mahout org.apache.mahout.clustering.syntheticcontrol."${clustertype}".Job ${options}
    fi
  else
    echo " HADOOP is not running. Please make sure you hadoop is running. "
  fi
elif [ "$MAHOUT_LOCAL" != "" ]; then
  echo "running MAHOUT_LOCAL"
  cp ${WORK_DIR}/synthetic_control.data testdata
  ../../bin/mahout org.apache.mahout.clustering.syntheticcontrol."${clustertype}".Job
  rm testdata
else
  echo " HADOOP_HOME variable is not set. Please set this environment variable and rerun the script"
fi
# Remove the work directory
rm -rf ${WORK_DIR}
