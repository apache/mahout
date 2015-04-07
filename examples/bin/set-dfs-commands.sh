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
# Requires $HADOOP_HOME to be set.
#
# Figures out the major version of Hadoop we're using and sets commands
# for dfs commands
#
# Run by each example script.

# Find a hadoop shell
if [ "$HADOOP_HOME" != "" ] && [ "$MAHOUT_LOCAL" == "" ] ; then
  HADOOP="${HADOOP_HOME}/bin/hadoop"
  if [ ! -e $HADOOP ]; then
    echo "Can't find hadoop in $HADOOP, exiting"
    exit 1
  fi
fi

# Check Hadoop version
v=`${HADOOP_HOME}/bin/hadoop version | egrep "Hadoop [0-9]+.[0-9]+.[0-9]+" | cut -f 2 -d ' ' | cut -f 1 -d '.'`

if [ $v -eq "1" -o $v -eq "0" ]
then
  echo "Discovered Hadoop v0 or v1."
  export DFS="${HADOOP_HOME}/bin/hadoop dfs"
  export DFSRM="$DFS -rmr -skipTrash"
elif [ $v -eq "2" ]
then
  echo "Discovered Hadoop v2."
  export DFS="${HADOOP_HOME}/bin/hdfs dfs"
  export DFSRM="$DFS -rm -r -skipTrash"
else
  echo "Can't determine Hadoop version."
  exit 1
fi
echo "Setting dfs command to $DFS, dfs rm to $DFSRM."

export HVERSION=$v 
