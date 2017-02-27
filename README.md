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

####Setting up your Environment
Whether you are using Mahout's Shell, running command line jobs or using it as a library to build your own apps you'll need to setup several environment variables. Edit your environment in `~/.bash_profile` for Mac or `~/.bashrc` for many linux distributions. Add the following
```
export MAHOUT_HOME=/path/to/mahout
export MAHOUT_LOCAL=true # for running standalone on your dev machine, 
# unset MAHOUT_LOCAL for running on a cluster
```
You will need a `$JAVA_HOME`, and if you are running on Spark, you will also need `$SPARK_HOME`

Note when running the spark-shell job it can help to set some JVM options so you don't run out of memory:
```
$MAHOUT_OPTS="-Xmx6g -XX:MaxPermSize=512m" mahout spark-shell
```

####Using Mahout as a Library
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
```
<dependency>
    <groupId>org.apache.mahout</groupId>
    <artifactId>mahout-flink_2.10</artifactId>
    <version>${mahout.version}</version>
</dependency>
```

Note that due to an intermittent out-of-memory bug in a Flink test we have disabled it from the binary releases. To use Flink please uncomment the line in the `<modules>` block so it reads `<module>flink</module>`.

####Examples
For examples of how to use Mahout, see the examples directory located in `examples/bin`


For information on how to contribute, visit the [How to Contribute Page](https://mahout.apache.org/developers/how-to-contribute.html)
  

####Legal
Please see the `NOTICE.txt` included in this directory for more information.

[![Build Status](https://api.travis-ci.org/apache/mahout.svg?branch=master)](https://travis-ci.org/apache/mahout)
[![Coverage Status](https://coveralls.io/repos/github/apache/mahout/badge.svg?branch=master)](https://coveralls.io/github/apache/mahout?branch=master)