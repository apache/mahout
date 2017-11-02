#!/bin/bash

#Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Assuming AWS Base AMI (RHEL/CentOS based)
# This script also assumes being run by root

yum install -y tmux
yum install -y htop

#Maven first
wget https://archive.apache.org/dist/maven/maven-3/3.3.9/binaries/apache-maven-3.3.9-bin.zip
unzip -qq apache-maven-3.3.9-bin.zip
export M2_HOME=$PWD/apache-maven-3.3.9
export PATH=$M2_HOME/bin:$PATH


# ViennaCL 
# setup on master
# yum install -y ocl-icd-libopencl1
# wget https://github.com/viennacl/viennacl-dev/archive/release-1.7.1.zip
# unzip -qq release-1.7.1.zip
# cp -r viennacl-dev-release-1.7.1/viennacl /usr/include/viennacl
# cp -r viennacl-dev-release-1.7.1/CL /usr/include/C

# Mahout 0.13.0 next (assuming no GPU here)
git clone https://github.com/apache/mahout.git
cd mahout
git fetch --all --tags --prune
git checkout tags/0.13.0
mvn clean package -Pviennacl-omp,hadoop2 -DskipTests
cd ..

#JCuda
#wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-rhel6-9-0-local-9.0.176-1.x86_64-rpm

#Setup on master
#rpm -i cuda-repo-rhel6-9-0-local-9.0.176-1.x86_64.rpm
#yum clean all
#yum install cuda

#setup JCuda on slaves
#for slave in spark*/conf/slaves; do
#    ssh -t $slave rpm -i cuda-repo-rhel6-9-0-local-9.0.176-1.x86_64.rpm
#   ssh -t $slave yum clean all
#   ssh -t $slave yum install cuda
#done

#Copy down zeppelin to master
wget http://mirrors.ocf.berkeley.edu/apache/zeppelin/zeppelin-0.7.3/zeppelin-0.7.3-bin-all.tgz
tar -xzvf zeppelin-0.7.3-bin-all.tgz

#Start zeppelin server:
#  TODO

#Start Spark Server
spark*/sbin/start-all.sh
