

#### Instructions for Installing Docker


Do you need super user privlidges to do docker stuff?

Try this:
`docker ps`

If you get an error that looks like this:
```
```

Then try `sudo docker ps`
Should get something that looks like this
```
```


#### Start Dockers


`mvn -PdockerITs docker:start`

After which run
`docker ps`

```
CONTAINER ID        IMAGE                            COMMAND                  CREATED              STATUS              PORTS                                                                                                                                NAMES
214e588b7233        sequenceiq/hadoop-docker:2.7.0   "/etc/bootstrap.sh -d"   About a minute ago   Up About a minute   2122/tcp, 8030-8033/tcp, 8040/tcp, 8042/tcp, 8088/tcp, 19888/tcp, 49707/tcp, 50010/tcp, 50020/tcp, 50070/tcp, 50075/tcp, 50090/tcp   sleepy_yalow
```

Note the `CONTAINER_ID` in this case (`214e588b7233`)

Now run
`docker exec -it 214e588b7233  bash`

You're into the Hadoop Cluster

now `cd /opt/mahout`

`exit` to leave the Docker image

`sudo mvn -PdockerITs docker:stop` to stop container

Stop All containers
`sudo docker stop $(sudo docker ps -a -q)`

`set-dfs-commands.sh` (after that classify-20newsgroups.sh started to work on 5)



MR Test

`./classify-20newsgroups.sh `
Option 1 (or maybe 2, i forget)

Takes a long time to 'upload' to hdfs- 6+ minutes.

Check output:
`$HADOOP_HOME/bin/hadoop dfs -ls /tmp/mahout-work-`
`/tmp/mahout-work-/20news-train-vectors/*` is the output

DOwnload spark (this is wrong)
```
cd /opt
yum install wget
wget http://d3kbcqa49mib13.cloudfront.net/spark-1.6.3-bin-hadoop2.4.tgz
tar -xzf spark-1.6.3-bin-hadoop2.4.tgz
export SPARK_HOME=/opt/spark-1.6.3-bin-hadoop2.4
spark-1.6.3-bin-hadoop2.4/sbin/start-all.sh
export MASTER=spark://$HOSTNAME:7077
cd mahout/examples/bin
./classify-20newsgroups
```
Option 3 (or 4)


#### Spark Item Similarity Test

Prototype, but wrong way to do this (violates principle of dockering)

```
mkdir /data
cd /data
wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
cd /opt/mahout/bin
12:56:57 INFO ShutdownHookManager: Deleting directory /tmp/spark-7c2e5d99-adbf-48dc-93c3-55f08cca19a8
$HADOOP_HOME/bin/hdfs dfs -put /data/ml-latest-small/ratings.csv /tmp/ratings.csv
./mahout spark-itemsimilarity --master spark://$HOSTNAME:7077 --input /tmp/ratings.csv --output /tmp/ratings_output --itemIDColumn 1 --rowIDColumn 0 --sparkExecutorMem 6g
```


Get an RC for testing
```
cd /opt
https://repository.apache.org/content/repositories/orgapachemahout-1034/org/apache/mahout/apache-mahout-distribution/0.13.0/apache-mahout-distribution-0.13.0.tar.gz
tar -xzf apache-mahout-distribution-0.13.0.tar.gz
sudo ln -s apache-mahout-distribution-0.13.0 mahout-rc
export MAHOUT_HOME=/opt/mahout-rc
cd mahout-rc/bin
```

/opt/mahout/src/conf:
/usr/local/hadoop/etc/hadoop:
/usr/java/default/lib/tools.jar:
/opt/mahout/mahout-*.jar:
/opt/mahout/math-scala/target/mahout-math-scala_2.10-0.13.0-SNAPSHOT-tests.jar:
/opt/mahout/math-scala/target/mahout-math-scala_2.10-0.13.0-SNAPSHOT.jar:
/opt/mahout/hdfs/target/mahout-hdfs-0.13.0-SNAPSHOT-tests.jar:
/opt/mahout/hdfs/target/mahout-hdfs-0.13.0-SNAPSHOT.jar:
/opt/mahout/math/target/mahout-math-0.13.0-SNAPSHOT-tests.jar:
/opt/mahout/math/target/mahout-math-0.13.0-SNAPSHOT.jar:
/opt/mahout/spark/target/mahout-spark_2.10-0.13.0-SNAPSHOT-dependency-reduced.jar:
/opt/mahout/spark/target/mahout-spark_2.10-0.13.0-SNAPSHOT-tests.jar:
/opt/mahout/spark/target/mahout-spark_2.10-0.13.0-SNAPSHOT.jar:
/opt/mahout/spark-shell/target/mahout-spark-shell_*.jar:
/opt/mahout/viennacl/target/mahout-native-viennacl_2.10-0.13.0-SNAPSHOT-tests.jar:
/opt/mahout/viennacl/target/mahout-native-viennacl_2.10-0.13.0-SNAPSHOT.jar:
/opt/mahout/viennacl-omp/target/mahout-native-viennacl-omp_2.10-0.13.0-SNAPSHOT-tests.jar:/opt/mahout/viennacl-omp/target/mahout-native-viennacl-omp_2.10-0.13.0-SNAPSHOT.jar:/opt/mahout/viennacl-omp/target/mahout-native-viennacl-omp_2.10-0.13.0-SNAPSHOT-tests.jar:/opt/mahout/viennacl-omp/target/mahout-native-viennacl-omp_2.10-0.13.0-SNAPSHOT.jar:/opt/spark-1.6.3-bin-hadoop2.4/conf:/opt/spark-1.6.3-bin-hadoop2.4/lib/spark-assembly-1.6.3-hadoop2.4.0.jar:/opt/spark-1.6.3-bin-hadoop2.4/lib/datanucleus-rdbms-3.2.9.jar:/opt/spark-1.6.3-bin-hadoop2.4/lib/datanucleus-api-jdo-3.2.6.jar:/opt/spark-1.6.3-bin-hadoop2.4/lib/datanucleus-core-3.2.10.jar:/usr/local/hadoop/etc/hadoop:/opt/mahout/bin/mahout-spark-class.sh:/opt/mahout/viennacl/target/mahout-native-viennacl_2.10-0.13.0-SNAPSHOT-tests.jar:/opt/mahout/viennacl/target/mahout-native-viennacl_2.10-0.13.0-SNAPSHOT.jar:/opt/mahout/viennacl-omp/target/mahout-native-viennacl-omp_2.10-0.13.0-SNAPSHOT-tests.jar:/opt/mahout/viennacl-omp/target/mahout-native-viennacl-omp_2.10-0.13.0-SNAPSHOT.jar:/opt/mahout/lib/*.jar


$SPARK_HOME/bin/spark-shell --jars "/opt/mahout/math-scala/target/mahout-math-scala_2.10-0.13.0-SNAPSHOT.jar,/opt/mahout/math/target/mahout-math-0.13.0-SNAPSHOT.jar,/opt/mahout/spark/target/mahout-spark_2.10-0.13.0-SNAPSHOT.jar,/opt/mahout/spark/target/mahout-spark_2.10-0.13.0-SNAPSHOT-dependency-reduced.jar" -i $MAHOUT_HOME/bin/load-shell.scala --conf spark.kryo.referenceTracking=false --conf spark.kryo.registrator=org.apache.mahout.sparkbindings.io.MahoutKryoRegistrator --conf spark.kryoserializer.buffer=32k --conf spark.kryoserializer.buffer.max=600m --conf spark.serializer=org.apache.spark.serializer.KryoSerializer $@