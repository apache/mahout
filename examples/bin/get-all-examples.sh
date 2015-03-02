#!/usr/bin/env bash
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

# Clones Mahout example code from remote repositories with their own 
# build process.  Follow the README for each example for instructions.
#
# Usage:  change into the mahout directory and type:
#  examples/bin/get-all-examples.sh

# Solr-recommender
echo " Solr-recommender example: "
echo " 1) imports text 'log files' of some delimited form for user preferences"
echo " 2) creates the correct Mahout files and stores distionaries to translate external Id to and from Mahout Ids"
echo " 3) it implements a prototype two actions 'cross-recommender', which takes two actions made by the same user and creates recommendations"
echo " 4) it creates output for user->preference history CSV and and item->similar items 'similarity' matrix for use in a Solr-recommender."
echo "    To use Solr you would index the similarity matrix CSV, and use user preference history from the history CSV as a query, the result"
echo "    from Solr will be an ordered list of recommendations returning the same item Ids as were input."
echo " For further description see the README.md here https://github.com/pferrel/solr-recommender"
echo " To build run 'cd solr-recommender; mvn install'"
echo " To process the example after building make sure MAHOUT_LOCAL IS SET and hadoop is in local mode then "
echo " run 'cd scripts; ./solr-recommender-example'"
git clone https://github.com/pferrel/solr-recommender
