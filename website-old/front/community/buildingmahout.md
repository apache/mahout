---
layout: default
title: Building Mahout
theme: 
    name: mahout2
---


# Building Mahout from Source

## Prerequisites

* Java JDK 1.7
* Apache Maven 3.3.9


## Getting the source code

Checkout the sources from the [Mahout GitHub repository](https://github.com/apache/mahout)
either via
 
    git clone git@github.com:apache/mahout.git
or
 
    git clone https://github.com/apache/mahout.git

## Building From Source

###### Prerequisites:

Linux Environment (preferably Ubuntu 16.04.x) Note: Currently only the JVM-only build will work on a Mac.
gcc > 4.x
NVIDIA Card (installed with OpenCL drivers alongside usual GPU drivers)

###### Downloads

Install java 1.7+ in an easily accessible directory (for this example,  ~/java/)
http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html
    
Create a directory ~/apache/ .
    
Download apache Maven 3.3.9 and un-tar/gunzip to ~/apache/apache-maven-3.3.9/ .
https://maven.apache.org/download.cgi
        
Download and un-tar/gunzip Hadoop 2.4.1 to ~/apache/hadoop-2.4.1/ .
https://archive.apache.org/dist/hadoop/common/hadoop-2.4.1/    

Download and un-tar/gunzip spark-1.6.3-bin-hadoop2.4 to  ~/apache/ .
http://spark.apache.org/downloads.html
Choose release: Spark-1.6.3 (Nov 07 2016)
Choose package type: Pre-Built for Hadoop 2.4

Install ViennaCL 1.7.0+
If running Ubuntu 16.04+

```
sudo apt-get install libviennacl-dev
```

Otherwise if your distribution’s package manager does not have a viennniacl-dev package >1.7.0, clone it directly into the directory which will be included in when  being compiled by Mahout:

```
mkdir ~/tmp
cd ~/tmp && git clone https://github.com/viennacl/viennacl-dev.git
cp -r viennacl/ /usr/local/
cp -r CL/ /usr/local/
```

Ensure that the OpenCL 1.2+ drivers are installed (packed with most consumer grade NVIDIA drivers).  Not sure about higher end cards.

Clone mahout repository into `~/apache`.

```
git clone https://github.com/apache/mahout.git
```

###### Configuration

When building mahout for a spark backend, we need four System Environment variables set:
```
    export MAHOUT_HOME=/home/<user>/apache/mahout
    export HADOOP_HOME=/home/<user>/apache/hadoop-2.4.1
    export SPARK_HOME=/home/<user>/apache/spark-1.6.3-bin-hadoop2.4    
    export JAVA_HOME=/home/<user>/java/jdk-1.8.121
```

Mahout on Spark regularly uses one more env variable, the IP of the Spark cluster’s master node (usually the node which one would be logged into).

To use 4 local cores (Spark master need not be running)
```
export MASTER=local[4]
```
To use all available local cores (again, Spark master need not be running)
```
export MASTER=local[*]
```
To point to a cluster with spark running: 
```
export MASTER=spark://master.ip.address:7077
```

We then add these to the path:

```
   PATH=$PATH$:MAHOUT_HOME/bin:$HADOOP_HOME/bin:$SPARK_HOME/bin:$JAVA_HOME/bin
```

These should be added to the your ~/.bashrc file.


###### Building Mahout with Apache Maven

Currently Mahout has 3 builds.  From the  $MAHOUT_HOME directory we may issue the commands to build each using mvn profiles.

JVM only:
```
mvn clean install -DskipTests
```

JVM with native OpenMP level 2 and level 3 matrix/vector Multiplication
```
mvn clean install -Pviennacl-omp -Phadoop2 -DskipTests
```
JVM with native OpenMP and OpenCL for Level 2 and level 3 matrix/vector Multiplication.  (GPU errors fall back to OpenMP, currently only a single GPU/node is supported).
```
mvn clean install -Pviennacl -Phadoop2 -DskipTests
```
 