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
layout: doc-page
title: Mahout Samsara

    
---
# FAQ for using Mahout with Spark

**Q: Mahout Spark shell doesn't start; "ClassNotFound" problems or various classpath problems.**

**A:** So far as of the time of this writing all reported problems starting the Spark shell in Mahout were revolving 
around classpath issues one way or another. 

If you are getting method signature like errors, most probably you have mismatch between Mahout's Spark dependency 
and actual Spark installed. (At the time of this writing the HEAD depends on Spark 1.1.0) but check mahout/pom.xml.

Troubleshooting general classpath issues is pretty straightforward. Since Mahout is using Spark's installation 
and its classpath as reported by Spark itself for Spark-related dependencies, it is important to make sure 
the classpath is sane and is made available to Mahout:

1. Check Spark is of correct version (same as in Mahout's poms), is compiled and SPARK_HOME is set.
2. Check Mahout is compiled and MAHOUT_HOME is set.
3. Run `$SPARK_HOME/bin/compute-classpath.sh` and make sure it produces sane result with no errors. 
If it outputs something other than a straightforward classpath string, most likely Spark is not compiled/set correctly (later spark versions require 
`sbt/sbt assembly` to be run, simply runnig `sbt/sbt publish-local` is not enough any longer).
4. Run `$MAHOUT_HOME/bin/mahout -spark classpath` and check that path reported in step (3) is included.

**Q: I am using the command line Mahout jobs that run on Spark or am writing my own application that uses 
Mahout's Spark code. When I run the code on my cluster I get ClassNotFound or signature errors during serialization. 
What's wrong?**
 
**A:** The Spark artifacts in the maven ecosystem may not match the exact binary you are running on your cluster. This may 
cause class name or version mismatches. In this case you may wish 
to build Spark yourself to guarantee that you are running exactly what you are building Mahout against. To do this follow these steps
in order:

1. Build Spark with maven, but **do not** use the "package" target as described on the Spark site. Build with the "clean install" target instead. 
Something like: "mvn clean install -Dhadoop1.2.1" or whatever your particular build options are. This will put the jars for Spark
in the local maven cache.
2. Deploy **your** Spark build to your cluster and test it there.
3. Build Mahout. This will cause maven to pull the jars for Spark from the local maven cache and may resolve missing 
or mis-identified classes.
4. if you are building your own code do so against the local builds of Spark and Mahout.

**Q: The implicit SparkContext 'sc' does not work in the Mahout spark-shell.**

**A:** In the Mahout spark-shell the SparkContext is called 'sdc', where the 'd' stands for distributed. 




