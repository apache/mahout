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

### Profiles Reference

Apache Maven encourages users to make use of build profiles for selectively building modules.

The command 
```bash
mvn clean package
```

Is the basic build command. This default will build the following packages.

```bash
[INFO] Apache Mahout 
[INFO] Mahout Core 
[INFO] Mahout Engine 
[INFO] - Mahout HDFS Support 
[INFO] - Mahout Spark Engine 
[INFO] Mahout Community 
[INFO] - Mahout Spark CLI Drivers 
```

The following profiles are available for building optional components:
<center>
	<table>
		<tr>
			<th>Profile</th>
			<th colspan="5">Description</th>
		</tr>
		<tr>
			<td><code>all</code></td>
			<td>Build all modules</td>
		<tr>
			<td><code>apache-release</code></td>
			<td>Used for releasing Apache Mahout. See <a href="http://mahout.apache.org/developers/how-to-release">How To Release</a> for more information.</td>
		</tr>
		<tr>
			<td><code>flink-batch</code></td>
			<td>Build Community Engine of Mahout for Apache Flink (Batch)</td>
		</tr>
		<tr>
			<td><code>h2o</code></td>
			<td>Build Community Engine of Mahout for H2o</td>
		</tr>
		<tr>
			<td><code>mahout-mr</code></td>
			<td>Build Community maintained Mahout for Map Reduce</td>
		<tr>
			<td><code>viennacl</code></td>
			<td>Build Experimental ViennaCL (GPU) BLAS Pack</td>
		</tr>
		<tr>
			<td><code>viennacl-omp</code></td>
			<td>Build Experimental ViennaCL-OpenMP (CPU) BLAS Pack</td>
		<tr>
		<tr>
			<td><code>docs</code></td>
			<td>Profile for building Java/Scala Docs</td>
		<tr>
			<th>Mahout Specific Option</th>
			<th>Description</th>
		</tr>
		<tr>
			<td><code>skipCli</code></td>
			<td>Skip building the command line drivers for Mahout on Apache Spark</td>
		</tr>
	</table>
</center>

#### Example

If you want to build Apache Mahout with ViennaCL OpenMP support but skip the command line Spark drivers you would use this 
command to build:

```bash
mvn clean package -Pviennacl-omp -DskipCli
```

#### Building Java/Scala Docs

To build the Java/Scala docs use the maven `site` goal and the `docs` profile. 

Additionally, passing the `-Ddependency.locations.enabled=false` option will skip checking the dependency location and allow a much faster build.
  
```bash
mvn clean site -Pall,docs -Ddependency.locations.enabled=false
```



