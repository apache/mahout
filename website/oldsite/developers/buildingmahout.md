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

Compile Mahout using standard maven commands

    # With hadoop-2.4.1 dependency
    mvn clean compile

    # With hadoop-1.2.1 dependency
    mvn -Phadoop1 -Dhadoop.version=1.2.1 clean compile

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

 