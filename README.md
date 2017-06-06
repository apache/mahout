<!--
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to You under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

Welcome to Apache Mahout!
===========
The Apache Mahout™ project's goal is to build an environment for quickly creating scalable performant machine learning applications.

For additional information about Mahout, visit the [Mahout Home Page](http://mahout.apache.org/)

#### Setting up your Environment
Whether you are using Mahout's Shell, running command line jobs or using it as a library to build your own apps you'll need to setup several environment variables. Edit your environment in `~/.bash_profile` for Mac or `~/.bashrc` for many linux distributions. Add the following
```
export MAHOUT_HOME=/path/to/mahout
export MAHOUT_LOCAL=true # for running standalone on your dev machine, 
# unset MAHOUT_LOCAL for running on a cluster
```
You will need a `$JAVA_HOME`, and if you are running on Spark, you will also need `$SPARK_HOME`

#### Using Mahout as a Library
Running any application that uses Mahout will require installing a binary or source version and setting the environment.
To compile from source:
* `mvn -DskipTests clean install`
* To run tests do `mvn test`
* To set up your IDE, do `mvn eclipse:eclipse` or `mvn idea:idea`

To use maven, add the appropriate setting to your pom.xml or build.sbt following the template below.


To use the Samsara environment you'll need to include both the engine neutral math-scala dependency:
```
<dependency>
    <groupId>org.apache.mahout</groupId>
    <artifactId>mahout-math-scala_2.10</artifactId>
    <version>${mahout.version}</version>
</dependency>
```
and a dependency for back end engine translation, e.g:
```
<dependency>
    <groupId>org.apache.mahout</groupId>
    <artifactId>mahout-spark_2.10</artifactId>
    <version>${mahout.version}</version>
</dependency>
```
#### Building From Source

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

#### Testing the Mahout Environment

Mahout provides an extension to the spark-shell, which is good for getting to know the language, testing partition loads, prototyping algorithms, etc..

To launch the shell in local mode with 2 threads: simply do the following:
```
$ MASTER=local[2] mahout spark-shell
```

After a very verbose startup, a Mahout welcome screen will appear:

```
Loading /home/andy/sandbox/apache-mahout-distribution-0.13.0/bin/load-shell.scala...
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.sparkbindings._
sdc: org.apache.mahout.sparkbindings.SparkDistributedContext = org.apache.mahout.sparkbindings.SparkDistributedContext@3ca1f0a4

                _                 _
_ __ ___   __ _| |__   ___  _   _| |_
 '_ ` _ \ / _` | '_ \ / _ \| | | | __|
 | | | | (_| | | | | (_) | |_| | |_
_| |_| |_|\__,_|_| |_|\___/ \__,_|\__|  version 0.13.0


That file does not exist


scala>
```
At the scala> prompt, enter: 
```   
scala> :load /home/<andy>/apache/mahout/examples
                               /bin/SparseSparseDrmTimer.mscala
```
Which will load a matrix multiplication timer function definition. To run the matrix timer: 
```
        scala> timeSparseDRMMMul(1000,1000,1000,1,.02,1234L)
            {...} res3: Long = 16321
```
We can see that the JVM only version is rather slow, thus our motive for GPU and Native Multithreading support.

To get an idea of what’s going on under the hood of the timer, we may examine the .mscala (mahout scala) code which is both fully functional scala and the Mahout R-Like DSL for tensor algebra:    
```



def timeSparseDRMMMul(m: Int, n: Int, s: Int, para: Int, pctDense: Double = .20, seed: Long = 1234L): Long = {
  val drmA = drmParallelizeEmpty(m , s, para).mapBlock(){
       case (keys,block:Matrix) =>
           val R =  scala.util.Random
           R.setSeed(seed)
           val blockB = new SparseRowMatrix(block.nrow, block.ncol)
           blockB := {x => if (R.nextDouble < pctDense) R.nextDouble else x }
       (keys -> blockB)
  }

  val drmB = drmParallelizeEmpty(s , n, para).mapBlock(){
       case (keys,block:Matrix) =>
           val R =  scala.util.Random
           R.setSeed(seed + 1)
           val blockB = new SparseRowMatrix(block.nrow, block.ncol)
           blockB := {x => if (R.nextDouble < pctDense) R.nextDouble else x }
       (keys -> blockB)
  }

  var time = System.currentTimeMillis()

  val drmC = drmA %*% drmB
 
  // trigger computation
  drmC.numRows()

  time = System.currentTimeMillis() - time

  time  
 
}
```

For more information please see the following references:

http://mahout.apache.org/users/environment/in-core-reference.html

http://mahout.apache.org/users/environment/out-of-core-reference.html

http://mahout.apache.org/users/sparkbindings/play-with-shell.html

http://mahout.apache.org/users/environment/classify-a-doc-from-the-shell.html

Note that due to an intermittent out-of-memory bug in a Flink test we have disabled it from the binary releases. To use Flink please uncomment the line in the root pom.xml in the `<modules>` block so it reads `<module>flink</module>`.

#### Examples
For examples of how to use Mahout, see the examples directory located in `examples/bin`

For information on how to contribute, visit the [How to Contribute Page](https://mahout.apache.org/developers/how-to-contribute.html)

#### Legal
Please see the `NOTICE.txt` included in this directory for more information.

[![Build Status](https://api.travis-ci.org/apache/mahout.svg?branch=master)](https://travis-ci.org/apache/mahout)
<!--
[![Coverage Status](https://coveralls.io/repos/github/apache/mahout/badge.svg?branch=master)](https://coveralls.io/github/apache/mahout?branch=master)
-->
