#!/usr/bin/env bash


$FLINK_HOME/bin/flink run \
 /home/rawkintrevo/gits/cylons/flink-engine/target/flink-engine-1.0-SNAPSHOT.jar \
 -c org.apache.mahout.cylon-example.flinkengine.apps.VectorsAndCoords \
 -i http://bglive-a.bitgravity.com/ndtv/247hi/live/native -s http://localhost:8983/solr/cylonfaces -c /home/rawkintrevo/gits/opencv/data/haarcascades/haarcascade_frontalface_alt.xml -e /home/rawkintrevo/gits/cylons/data/eigenfaces-130_2.11/eigenfaces.mmat -m /home/rawkintrevo/gits/cylons/data/eigenfaces-130_2.11/colMeans.mmat