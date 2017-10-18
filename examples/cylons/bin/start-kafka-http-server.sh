#!/usr/bin/env bash


echo "To test raw Kafka Producer surf to http://localhost:8090/cylon/cam/test/test"
echo "To test Flink Markup Kafka Producer surf to http://localhost:8090/cylon/cam/flink-test/test"
java -Dfile.encoding=UTF-8 -jar ./http-server-1.0-SNAPSHOT-jar-with-dependencies.jar -p 8090


