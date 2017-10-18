#!/usr/bin/env bash


java -cp $CYLON_HOME/examples/target/examples-1.0-SNAPSHOT-jar-with-dependencies.jar \
 org.rawkintrevo.cylon.examples.FacesToVectorsApp \
 -c $OPEN_CV/data/haarcascades/haarcascade_frontalface_default.xml \
	-e $CYLON_HOME/data/eigenfaces-130_2.11/eigenfaces.mmat \
	-m $CYLON_HOME/data/eigenfaces-130_2.11/colMeans.mmat \
	-s http://localhost:8983/solr/cylonfaces \
	-i rtsp://192.168.100.1:554/cam1/mpeg4
	-f 1800

	# 90,000 fps is "full speed" so take that x {speed you want}% e.g. 10% of all frames -> 9000

