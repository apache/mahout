---
layout: page
title: Building Mahout

    
---


# Building Mahout from Source

## Prerequisites

* Java JDK 1.7
* Apache Maven 3.3.9

<a name="getting-code"></a>
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

From the  $MAHOUT_HOME directory we may issue the commands to build each using mvn profiles.

JVM only:
```
mvn clean install -DskipTests
```

JVM with native OpenMP level 2 and level 3 matrix/vector Multiplication
```
mvn clean install -Pviennacl-omp -DskipTests
```
JVM with native OpenMP and OpenCL for Level 2 and level 3 matrix/vector Multiplication.  (GPU errors fall back to OpenMP, currently only a single GPU/node is supported).
```
mvn clean install -Pviennacl -DskipTests
```

### Changing Scala Version


A convenience script for updating maven dependencies is included in `buildtools`

```bash
cd $MAHOUT_HOME/buildtools
./change-scala-version.sh 2.11
```

Now go back to `$MAHOUT_HOME` and execute

```bash
mvn clean install 
```


### The Distribution Profile

The distribution profile, among other things, will produce the same artifact for multiple Scala and Spark versions.

Specifically, in addition to creating all of the

Default Targets:
- Spark 1.6 Bindings, Scala-2.10
- Mahout-Math Scala-2.10
- ViennaCL Scala-2.10*
- ViennaCL-OMP Scala-2.10*
- H2O Scala-2.10

It will also create:
- Spark 2.0 Bindings, Scala-2.11
- Spark 2.1 Bindings, Scala-2.11
- Mahout-Math Scala-2.11
- ViennaCL Scala-2.11*
- ViennaCL-OMP Scala-2.11*
- H2O Scala-2.11

Note: * ViennaCLs are only created if the `viennacl` or `viennacl-omp` profiles are activated.

By default, this phase will execute the `package` lifecycle goal on all built "extra" varients.

E.g. if you were to run

```bash
mvn clean install -Pdistribution
```

You will `install` all of the "Default Targets" but only `package` the "Also created".

If you wish to `install` all of the above, you can set the `lifecycle.target` switch as follows:

```bash
mvn clean install -Pdistribution -Dlifecycle.target=install
```



