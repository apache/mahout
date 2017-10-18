
# Quickstart

It goes without saying you should be running Linux, like a grown up. 

I'm on Ubuntu 17.04 for whatever that is worth.


### Environment Variables which need to be set

- `CYLON_HOME` -  e.g. `$MAHOUT_HOME/examples/cylons`
- `OPEN_CV` - e.g. `/path/to/your/opencv3` see build OpenCV section


### Build OpenCV 3.3.0

Helpful links.

[K.M.A.G. Y.O.Y.O](http://www.urbandictionary.com/define.php?term=KMAG%20YOYO)

[May the odds forever be in your favor](http://opencv-java-tutorials.readthedocs.io/en/latest/01-installing-opencv-for-java.html)

Set `OPEN_CV` env variable 
	
	export OPEN_CV=/path/to/your/opencv

Here's a recipe that may be for success or doom?
	
	git clone http://github.com/opencv/opencv
	cd opencv
	mkdir build
	cd build
	cmake ../
  
and once that all works
  
	cd ../
	cd platforms
	cd maven
	mvn clean install -DskipTests


### Build Cylon

	git clone https://github.com/rawkintrevo/cylons
	export CYLON_HOME=/path/to/cylons
	cd $CYLON_HOME
	mvn clean package

### Start a Solr in Docker 

**Install Docker** _if needed_.

	sudo apt install docker.io
	
**Start Solr in Docker**

	sudo docker run --name cylon_solr -d -p 8983:8983 -t solr
	
**Create core** `cylonfaces`

	sudo docker exec -it --user=solr cylon_solr bin/solr create_core -c cylonfaces
	
### Run Local

Make sure to set CYLON_HOME to the directory where you cloned this, OPEN_CV to the directory where
you built OpenCV 3.x and Solr is running on localhost:8983 with the collection 'cylonfaces' available.




### Full Blown (not working quite right atm)

### Start Apache Kafka 

`$KAFKA_HOME/bin/zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties`

*Hint: Change set* `log.retention.minutes=1` *in* `$KAFKA_HOME/config/server.properties` 

`$KAFKA_HOME/bin/kafka-server-start.sh $KAFKA_HOME/config/server.properties`


### Start Apache Flink

`$FLINK_HOME/bin/start-cluster.sh`

(You can now go to [http://localhost:8080](http://localhost:8080) to see the Flink GUI, if you didn't already know that.)

# Run Sample Scripts

### Testing Kafka Video Producer

`$CYLON_HOME/bin/test-video-producer.sh`

The test will connect to an RSTP video stream, which is the same type of video that the webcam generates.

Drones get about 8 minutes each run, and there is some other quirks to connecting to them, we test our video/markup/face
 detection ability with this video feed.
 
### Viewing the Output

Start the http-server with 

`$CYLON_HOME/bin/test-http-server.sh`

The video server URLs are 

`http://localhost:8090/cylon/cam/<topic>/<key>`

So to check the test feed that was just set up

[http://localhost:8090/cylon/cam/test/test](http://localhost:8090/cylon/cam/test/test)

### Flink

Copy OpenCV jar and binary to Flink Libraries Folder as well as the static loader

	cp $OPENCV_HOME/build/bin/opencv-330.jar $FLINK_HOME/lib
	cp $OPENCV_HOME/build/lib/libopencv_java330.so $FLINK_HOME/lib
	cp $CYLON_HOME/opencv/target/opencv-1.0-SNAPSHOT.jar $FLINK_HOME/lib

	
This is required to avoid a variety of experiences of the dreaded
`java.lang.UnsatisfiedLinkError`
	
Now you can run the Flink face detection demo (which marks up with detected faces)

`$CYLON_HOME/bin/test-flink-faces.sh`

# Observer

You should be able to see some interesting things at:

[http://localhost:8090/cylon/cam/test/test](http://localhost:8090/cylon/cam/test/test)

[http://localhost:8090/cylon/cam/test-flink/test](http://localhost:8090/cylon/cam/test-flink/test)

