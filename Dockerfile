#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


FROM openjdk:8-alpine

ARG spark_uid=185

# Before building the mahout docker image, we must build a spark distrobution following
# the instructions in http://spark.apache.org/docs/latest/building-spark.html.
# this Dockerfile will build Spark version 2.4.4 against Scala 2.12.
# docker build -t mahout:latest -f resource_managers/docker/kubernetes/src/main/dockerfiles/Dockerfile .


RUN set -ex && \
    apk upgrade --no-cache && \
    ln -s /lib /lib64 && \
    apk add --no-cache bash tini libc6-compat linux-pam krb5 krb5-libs nss curl openssl && \
    mkdir -p /opt/mahout && \
    mkdir -p /opt/mahout/examples && \
    mkdir -p /opt/mahout/work-dir && \
    mkdir -p /opt/spark && \
    export MAHOUT_DOCKER_HOME=/opt/mahout && \
    export SPARK_VERSION=spark-2.4.4
    export SPARK_BASE=/opt/spark && \
    export SPARK_HOME=${SPARK_BASE}/${SPARK_VERSION}
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m" && \
    export SPARK_SRC_URL="https://archive.apache.org/dist/spark/${SPARK_VERSION}/${SPARK_VERSION}.tgz" && \
    export SPARK_SRC_SHA512_URL="https://archive.apache.org/dist/spark/${SPARK_VERSION}/${SPARK_VERSION}.tgz.sha512" && \
    export SPARK_SRC_SHA512="D33096E7EFBC4B131004C85FB5833AC3BAB8F097644CBE68D89ADC81F5144B5535337FD0082FA04A19C2870BD7D84758E8AE9C6EC1C7F3DF9FED35325EEA8928" && \
    curl  -LfsS $SPARK_SRC_URL -o ${SPARK_BASE}/${SPARK_VERSION}.tgz  && \
    curl  -LfsS $SPARK_SRC_SHA512_URL -o ${SPARK_BASE}/${SPARK_VERSION}.tgz.sha512
    #$SPARK_HOME/$SPARK_VERSION.sha512 ${SPARK_HOME}/$SPARK_VERSION.tgz | shasum -a 512 -c - && \
    tar -xzvf ${SPARK_BASE}/${SPARK_VERSION}.tgz -C ${SPARK_BASE}/&& \
    echo ${SPARK_BASE}/${SPARK_VERSION}
    sh ${SPARK_HOME}/dev/change-scala-version.sh 2.12 && \
    sh ${SPARK_HOME}/build/mvn -Pkubernetes -Pscala-2.12 -DskipTests clean package
    touch /opt/mahout/RELEASE && \
    # below is for nodes.  for the moment lets get a master up
    # rm /bin/sh && \
    # ln -sv /bin/bash /bin/sh && \
    # echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    # chgrp root /etc/passwd && chmod ug+rw /etc/passwd


COPY lib /opt/mahout/lib
COPY bin /opt/mahout/bin
COPY resource-managers/kubernetes/docker/src/main/dockerfiles/entrypoint.sh /opt/
COPY examples /opt/mahout/examples

COPY spark-build/jars /opt/spark/jars
COPY spark-build/bin /opt/spark/bin
COPY spark-build/sbin /opt/spark/sbin
COPY spark-build/kubernetes/tests /opt/spark/tests
COPY spark-build/data /opt/spark/data

ENV MAHOUT_HOME /opt/mahout
ENV SPARK_HOME /opt/spark



WORKDIR /opt/mahout/work-dir
RUN chmod g+w /opt/mahout/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]

# Specify the User that the actual main process will run as
USER ${spark_uid}