#!/usr/bin/env bash

echo "Make sure to set CYLON_HOME to the directory where you cloned this, OPEN_CV to the directory where\
you built OpenCV 3.x and Solr is running on localhost:8983 with the collection 'cylonfaces' available"

## In this Example Facial Recognition is all done locally.

java -cp $CYLON_HOME/examples/target/examples-1.0-SNAPSHOT-jar-with-dependencies.jar \
 org.apache.mahout.cylon-example.examples.localfacialrecognition.SimpleFacialRecognitionApp \
 -c $OPEN_CV/data/haarcascades/haarcascade_frontalface_default.xml \
	-e $CYLON_HOME/data/eigenfaces-130_2.11/eigenfaces.mmat \
	-m $CYLON_HOME/data/eigenfaces-130_2.11/colMeans.mmat \
	-s http://localhost:8983/solr/cylonfaces \
	-i http://bglive-a.bitgravity.com/ndtv/247hi/live/native
