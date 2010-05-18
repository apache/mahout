#/**
# * Licensed to the Apache Software Foundation (ASF) under one or more
# * contributor license agreements.  See the NOTICE file distributed with
# * this work for additional information regarding copyright ownership.
# * The ASF licenses this file to You under the Apache License, Version 2.0
# * (the "License"); you may not use this file except in compliance with
# * the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

#
# Downloads the Reuters dataset and prepares it for clustering
#
# To run:  change into the mahout directory and type:
#  examples/bin/build-reuters.sh
#!/bin/sh

cd examples/bin/
mkdir -p work
if [ ! -e work/reuters-out ]; then
  if [ ! -e work/reuters-sgm ]; then
    if [ ! -f work/reuters21578.tar.gz ]; then
      echo "Downloading Reuters-21578"
      curl http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz  -o work/reuters21578.tar.gz
    fi
    mkdir -p work/reuters-sgm
    echo "Extracting..."
    cd work/reuters-sgm && tar xzf ../reuters21578.tar.gz && cd .. && cd ..
  fi
fi

cd ../..
./bin/mahout org.apache.lucene.benchmark.utils.ExtractReuters ./examples/bin/work/reuters-sgm/ ./examples/bin/work/reuters-out/
./bin/mahout seqdirectory -i ./examples/bin/work/reuters-out/ -o ./examples/bin/work/reuters-out-seqdir -c UTF-8
./bin/mahout seq2sparse -i ./examples/bin/work/reuters-out-seqdir/ -o ./examples/bin/work/reuters-out-seqdir-sparse
#./bin/mahout kmeans -i ./examples/bin/work/reuters-out-seqdir-sparse/tfidf/vectors/ -c ./examples/bin/work/clusters -o ./examples/bin/work/reuters-kmeans -k 20 -ow
#./bin/mahout lda -i ./examples/bin/work/reuters-out-seqdir-sparse/tfidf/vectors -o ./examples/bin/work/reuters-lda -k 20 -v 50000 -ow

