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
# Runs the LDA examples using Reuters.
#
# To run:  change into the mahout/examples directory (the parent of the one containing this file) and type:
#  bin/build-reuters.sh
#
#
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
  echo "Converting to plain text."
  mvn -e -q exec:java  -Dexec.mainClass="org.apache.lucene.benchmark.utils.ExtractReuters" -Dexec.args="work/reuters-sgm work/reuters-out" || exit
fi
# Create index
if [ ! -e work/index ]; then
  echo "Creating index";
  mvn -e exec:java -Dexec.classpathScope="test" -Dexec.mainClass="org.apache.lucene.benchmark.byTask.Benchmark" -Dexec.args="bin/lda.algorithm" || ( rm -rf work/index && exit )
fi
if [ ! -e work/vectors ]; then
  echo "Creating vectors from index"
  cd ../core
  mvn -q install -DskipTests=true
  cd ../utils/
  mvn -q compile
  mvn -e exec:java -Dexec.mainClass="org.apache.mahout.utils.vectors.lucene.Driver" \
    -Dexec.args="--dir ../examples/work/index/ --field body --dictOut ../examples/work/dict.txt \
    --output ../examples/work/vectors --minDF 100 --maxDFPercent 97" || exit
  cd ../core/
fi
echo "Running LDA"
rm -rf ../examples/work/lda
MAVEN_OPTS="-Xmx2G -ea" mvn -e exec:java -Dexec.mainClass=org.apache.mahout.clustering.lda.LDADriver -Dexec.args="-i ../examples/work/vectors -o ../examples/work/lda/\
  -k 20 -v 10000 --maxIter 40"
echo "Writing top words for each topic to to examples/work/topics/"
mvn -q exec:java -Dexec.mainClass="org.apache.mahout.clustering.lda.LDAPrintTopics" -Dexec.args="-i `ls -1dtr ../examples/work/lda/state-* | tail -1` -d ../examples/work/dict.txt -o ../examples/work/topics/ -w 100"
