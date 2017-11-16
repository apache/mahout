#!/usr/bin/env bash

$SPARK_HOME/bin/spark-submit \
  --class org.rawkintrevo.cylon.eigenfaces.CalcEigenfacesApp \
  --master spark://$HOSTNAME:7077 \
  $CYLON_HOME/eigenfaces/target/spark-eigenfaces-1.0-SNAPSHOT-jar-with-dependencies.jar \
  -o $CYLON_HOME/data/eigenfaces-30 \
  -k 30 -p 200
