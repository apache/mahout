---
layout: default
title: BuildingMahout
theme:
    name: retro-mahout
---

# Building Mahout from source

## Prerequisites

* Java JDK 1.7
* Apache Maven 3.3.3


## Getting the source code

Checkout the sources from the [Mahout GitHub repository](https://github.com/apache/mahout)
either via
 
    git clone git@github.com:apache/mahout.git
or
 
    git clone https://github.com/apache/mahout.git

##Hadoop version
Mahout code depends on hadoop-client artifact, with the default version 2.4.1. To build Mahout against to a
different hadoop version, hadoop.version property should be set accordingly and passed to the build command.
Hadoop1 clients would additionally require hadoop1 profile to be activated.

The build lifecycle is illustrated below. 

## Compiling

Compile Mahout using standard maven commands.  To view available profiles, see the pom.xml file in the root project directory.

    # With hadoop-2.4.1 dependency
    mvn clean compile

    # With hadoop-1.2.1 dependency
    mvn -Phadoop1 -Dhadoop.version=1.2.1 clean compile


    # With Spark 1.6 dependency (also supports Spark 2.0,2.1, and 2.2)
    mvn -Pspark-1.6 -DskipTests=true clean package


    # By default Spark 1.6,2.0,2.1 all use scala 2.10.  Spark-2.2 uses scala 2.11.  To override 
    # With Spark 1.6 dependency (also supports Spark-2.0,Spark-2.1,Spark-2.2)
    mvn -PSpark-1.6 -Pscala-2.11 -DskipTests=true clean package

    # Spark 2.2 requires java 8 and hadoop 2.6.  
    # For convenience, the default java and hadoop version settings are overridden in the pom.xml profile section.
    mvn -PSpark-2.2 clean package

##Packaging

Mahout has an extensive test suite which takes some time to run. If you just want to build Mahout, skip the tests like this

    # With hadoop-2.4.1 dependency
    mvn -DskipTests=true clean package

    # With hadoop-1.2.1 dependency
    mvn -Phadoop1 -Dhadoop.version=1.2.1 -DskipTests=true clean package


In order to add mahout artifact to your local repository, run

    # With hadoop-2.4.1 dependency
    mvn clean install

    # With hadoop-1.2.1 dependency
    mvn -Phadoop1 -Dhadoop.version=1.2.1 clean install

 