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

ENV spark_uid=185

ENV SCALA_MAJOR 2.12
ENV HADOOP_MAJOR 2.7
ENV SPARK_MAJOR_MINOR 2.4.4

# Before building the mahout docker image, we must build a spark distrobution following
# the instructions in http://spark.apache.org/docs/latest/building-spark.html.
# this Dockerfile will build Spark version 2.4.4 against Scala 2.12 by default.
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
    export SPARK_VERSION=spark-${SPARK_MAJOR_MINOR} && \
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
    sh ${SPARK_HOME}/dev/change-scala-version.sh ${SCALA_MAJOR} && \
    sh ${SPARK_HOME}/dev/make-distribution.sh --name ${DATE}-${REVISION} --pip --tgz -DzincPort=${ZINC_PORT} \
         -Phadoop-${HADOOP_MAJOR} -Pkubernetes -Pkinesis-asl -Phive -Phive-thriftserver

    #sh ${SPARK_HOME}/build/mvn -Pkubernetes -Pscala-${scala_version} -DskipTests clean package
    touch /opt/mahout/RELEASE && \
    # below is for nodes.  for the moment lets get a master up
    # rm /bin/sh && \
    # ln -sv /bin/bash /bin/sh && \
    # echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    # chgrp root /etc/passwd && chmod ug+rw /etc/passwd

ENV MAHOUT_HOME /opt/mahout
COPY lib ${MAHPOUT_HOME}/lib
COPY bin ${MAHPOUT_HOME}/bin
COPY resource-managers/kubernetes/docker/src/main/dockerfiles/entrypoint.sh ${MAHPOUT_HOME}
COPY resource-managers/kubernetes/docker/src/main/dockerfiles/Dockerfile ${MAHPOUT_HOME}
COPY examples ${MAHPOUT_HOME}/examples

ENV SPARK_HOME /opt/spark
COPY spark-build/jars ${SPARK_HOME}/jars
COPY spark-build/bin ${SPARK_HOME}/bin
COPY spark-build/sbin ${SPARK_HOME}/sbin
COPY spark-build/kubernetes/tests ${SPARK_HOME}/tests
COPY spark-build/data ${SPARK_HOME}/data

ENV MAHOUT_CLASSPATH ${MAHOUT_HOME}/lib
ENV SPARK_CLASSPATH ${SPARK_HOME}/jars

WORKDIR /opt/mahout/work-dir
RUN chmod g+w /opt/mahout/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]

# Specify the User that the actual main process will run as
USER ${spark_uid}