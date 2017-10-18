#!/usr/bin/env bash


java -cp $CYLON_HOME/examples/target/examples-1.0-SNAPSHOT-jar-with-dependencies.jar \
 org.rawkintrevo.cylon.examples.FacesToVectorsApp \
 -c $OPEN_CV/data/haarcascades/haarcascade_frontalface_default.xml \
	-e $CYLON_HOME/data/eigenfaces-130_2.11/eigenfaces.mmat \
	-m $CYLON_HOME/data/eigenfaces-130_2.11/colMeans.mmat \
	-s http://localhost:8983/solr/cylonfaces \
	-i http://bglive-a.bitgravity.com/ndtv/247hi/live/native
	-f 20
