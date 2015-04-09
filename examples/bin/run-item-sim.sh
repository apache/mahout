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
# To run:  change into the mahout directory and type:
# ./examples/bin/run-item-sim.sh
#
# This works only in Spark and Hadoop standalone mode for progress
# making this automatically sense the user's config see:
# https://issues.apache.org/jira/browse/MAHOUT-1679

OUT_DIR="/item-sim-out" # output dir for cooccurrence and cross-cooccurrence matrices
INPUT="/examples/src/main/resources/cf-data-purchase.txt" # purchase actions
INPUT2="/examples/src/main/resources/cf-data-view.txt" # view actions
FS=file://
PURCHASE=$FS$MAHOUT_HOME$INPUT
VIEW=$FS$MAHOUT_HOME$INPUT2
FS_OUPUT=$FS$MAHOUT_HOME$OUT_DIR
OUTPUT1=$MAHOUT$OUT_DIR/similarity-matrix/part-00000
OUTPUT2=$MAHOUT$OUT_DIR/cross-similarity-matrix/part-00000

#check to see if we can run with local fie system
if [$MAHOUT_HOME == ""]; then
  echo "MAHOUT_HOME is not set."
  exit
fi

#setup env
LOCAL=$MAHOUT_LOCAL # save state
export MAHOUT_LOCAL=true #set so the raw local file system is used

echo "To run:  change into the mahout directory and type:"
echo " ./examples/bin/run-item-sim.sh"
echo ""
echo "This runs spark-itemsimilarity on two user actions in two intput files"
echo "The example uses fake purchases and views of products, calculating"
echo "a cooccurrence indicator for purchase and a cross-cooccurrence indicator"
echo "for view (product detail view). The example is tiny so it can be followed"
echo "with a little intuition."
echo ""
echo "Note: This script only runs on a local installation of Spark and Mahout!"
echo "If you get 'file not found' errors you may have Spark running on Hadoop"
echo "To run this on a Spark + Hadoop cluster or pseudo-cluster do the following:"
echo ""
echo "  hadoop fs -put examples/src/main/resources/cf-* / "
echo "  mahout spark-itemsimilarity -i /cf-data-purchase.txt -i2 /cf-data-view.txt -o /item-sim-out \\"
echo "    -ma spark://your-spark-master:7077 -sem 4g"
echo ""
echo "Then look in /item-sim-out for output"
echo ""

# Remove old files
echo
echo "Removing old output file if it exists"
echo
rm -r $MAHOUT_HOME$OUT_DIR

mahout spark-itemsimilarity -i $PURCHASE -i2 $VIEW -o $FS_OUPUT -ma local

export MAHOUT_LOCAL=$LOCAL #restore state

echo "Look in " $FS_OUPUT " for spark-itemsimilarity indicator data."

echo ""
echo "Purchase cooccurrence indicators (itemid<tab>simliar items by purchase)"
echo ""
cat .$OUTPUT1
echo ""
echo "View cross-cooccurrence indicators (items<tab>similar items where views led to purchases)"
echo ""
cat .$OUTPUT2
echo ""
