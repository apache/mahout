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

OUTPUT="item-sim-out" # output dir for cooccurrence and cross-cooccurrence matrices
INPUT="examples/src/main/resources/cf-data-purchase.txt" # purchase actions
INPUT2="examples/src/main/resources/cf-data-view.txt" # view actions

#setup env
LOCAL=$MAHOUT_LOCAL # save state
export MAHOUT_LOCAL=true #set so the raw local file system is used

echo "This runs spark-itemsimilarity on two user actions in two intput files"
echo "The example uses fake purchases and views of products, calculating"
echo "a cooccurrence indicator for purchase and a cross-cooccurrence indicator"
echo "for view (product detail view). The example is tiny so it can be followed"
echo "with a little intuition."

# Remove old files
echo
echo "Removing old output file if it exists"
echo
rm -r $OUTPUT

mahout spark-itemsimilarity -i $INPUT -i2 $INPUT2 -o $OUTPUT -ma local

export MAHOUT_LOCAL=$LOCAL #restore state

echo "Look in " $OUTPUT " for spark-itemsimilarity indicator data."
