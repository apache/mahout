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
layout: default
title: 

   
---

# How to create and App using Mahout

This is an example of how to create a simple app using Mahout as a Library. The source is available on Github in the [3-input-cooc project](https://github.com/pferrel/3-input-cooc) with more explanation about what it does (has to do with collaborative filtering). For this tutorial we'll concentrate on the app rather than the data science.

The app reads in three user-item interactions types and creats indicators for them using cooccurrence and cross-cooccurrence. The indicators will be written to text files in a format ready for search engine indexing in search engine based recommender.

##Setup
In order to build and run the CooccurrenceDriver you need to install the following:

* Install the Java 7 JDK from Oracle. Mac users look here: [Java SE Development Kit 7u72](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html).
* Install sbt (simple build tool) 0.13.x for [Mac](http://www.scala-sbt.org/release/tutorial/Installing-sbt-on-Mac.html), [Linux](http://www.scala-sbt.org/release/tutorial/Installing-sbt-on-Linux.html) or [manual instalation](http://www.scala-sbt.org/release/tutorial/Manual-Installation.html).
* Install [Spark 1.1.1](https://spark.apache.org/docs/1.1.1/spark-standalone.html). Don't forget to setup SPARK_HOME
* Install [Mahout 0.10.0](http://mahout.apache.org/general/downloads.html). Don't forget to setup MAHOUT_HOME and MAHOUT_LOCAL

Why install if you are only using them as a library? Certain binaries and scripts are required by the libraries to get information about the environment like discovering where jars are located.

Spark requires a set of jars on the classpath for the client side part of an app and another set of jars must be passed to the Spark Context for running distributed code. The example should discover all the neccessary classes automatically.

##Application
Using Mahout as a library in an application will require a little Scala code. Scala has an App trait so we'll create an object, which inherits from ```App```


    object CooccurrenceDriver extends App {
    }
    

This will look a little different than Java since ```App``` does delayed initialization, which causes the body to be executed when the App is launched, just as in Java you would create a main method.

Before we can execute something on Spark we'll need to create a context. We could use raw Spark calls here but default values are setup for a Mahout context by using the Mahout helper function.

    implicit val mc = mahoutSparkContext(masterUrl = "local", 
      appName = "CooccurrenceDriver")
    
We need to read in three files containing different interaction types. The files will each be read into a Mahout IndexedDataset. This allows us to preserve application-specific user and item IDs throughout the calculations.

For example, here is data/purchase.csv:

    u1,iphone
    u1,ipad
    u2,nexus
    u2,galaxy
    u3,surface
    u4,iphone
    u4,galaxy

Mahout has a helper function that reads the text delimited files  SparkEngine.indexedDatasetDFSReadElements. The function reads single element tuples (user-id,item-id) in a distributed way to create the IndexedDataset. Distributed Row Matrices (DRM) and Vectors are important data types supplied by Mahout and IndexedDataset is like a very lightweight Dataframe in R, it wraps a DRM with HashBiMaps for row and column IDs. 

One important thing to note about this example is that we read in all datasets before we adjust the number of rows in them to match the total number of users in the data. This is so the math works out [(A'A, A'B, A'C)](http://mahout.apache.org/users/algorithms/intro-cooccurrence-spark.html) even if some users took one action but not another there must be the same number of rows in all matrices.

    /**
     * Read files of element tuples and create IndexedDatasets one per action. These 
     * share a userID BiMap but have their own itemID BiMaps
     */
    def readActions(actionInput: Array[(String, String)]): Array[(String, IndexedDataset)] = {
      var actions = Array[(String, IndexedDataset)]()

      val userDictionary: BiMap[String, Int] = HashBiMap.create()

      // The first action named in the sequence is the "primary" action and 
      // begins to fill up the user dictionary
      for ( actionDescription <- actionInput ) {// grab the path to actions
        val action: IndexedDataset = SparkEngine.indexedDatasetDFSReadElements(
          actionDescription._2,
          schema = DefaultIndexedDatasetElementReadSchema,
          existingRowIDs = userDictionary)
        userDictionary.putAll(action.rowIDs)
        // put the name in the tuple with the indexedDataset
        actions = actions :+ (actionDescription._1, action) 
      }

      // After all actions are read in the userDictonary will contain every user seen, 
      // even if they may not have taken all actions . Now we adjust the row rank of 
      // all IndxedDataset's to have this number of rows
      // Note: this is very important or the cooccurrence calc may fail
      val numUsers = userDictionary.size() // one more than the cardinality

      val resizedNameActionPairs = actions.map { a =>
        //resize the matrix by, in effect by adding empty rows
        val resizedMatrix = a._2.create(a._2.matrix, userDictionary, a._2.columnIDs).newRowCardinality(numUsers)
        (a._1, resizedMatrix) // return the Tuple of (name, IndexedDataset)
      }
      resizedNameActionPairs // return the array of Tuples
    }


Now that we have the data read in we can perform the cooccurrence calculation.

    // actions.map creates an array of just the IndeedDatasets
    val indicatorMatrices = SimilarityAnalysis.cooccurrencesIDSs(
      actions.map(a => a._2)) 

All we need to do now is write the indicators.

    // zip a pair of arrays into an array of pairs, reattaching the action names
    val indicatorDescriptions = actions.map(a => a._1).zip(indicatorMatrices)
    writeIndicators(indicatorDescriptions)


The ```writeIndicators``` method uses the default write function ```dfsWrite```.

    /**
     * Write indicatorMatrices to the output dir in the default format
     * for indexing by a search engine.
     */
    def writeIndicators( indicators: Array[(String, IndexedDataset)]) = {
      for (indicator <- indicators ) {
        // create a name based on the type of indicator
        val indicatorDir = OutputPath + indicator._1
        indicator._2.dfsWrite(
          indicatorDir,
          // Schema tells the writer to omit LLR strengths 
          // and format for search engine indexing
          IndexedDatasetWriteBooleanSchema) 
      }
    }
 

See the Github project for the full source. Now we create a build.sbt to build the example. 

    name := "cooccurrence-driver"

    organization := "com.finderbots"

    version := "0.1"

    scalaVersion := "2.10.4"

    val sparkVersion = "1.1.1"

    libraryDependencies ++= Seq(
      "log4j" % "log4j" % "1.2.17",
      // Mahout's Spark code
      "commons-io" % "commons-io" % "2.4",
      "org.apache.mahout" % "mahout-math-scala_2.10" % "0.10.0",
      "org.apache.mahout" % "mahout-spark_2.10" % "0.10.0",
      "org.apache.mahout" % "mahout-math" % "0.10.0",
      "org.apache.mahout" % "mahout-hdfs" % "0.10.0",
      // Google collections, AKA Guava
      "com.google.guava" % "guava" % "16.0")

    resolvers += "typesafe repo" at " http://repo.typesafe.com/typesafe/releases/"

    resolvers += Resolver.mavenLocal

    packSettings

    packMain := Map(
      "cooc" -> "CooccurrenceDriver")


##Build
Building the examples from project's root folder:

    $ sbt pack

This will automatically set up some launcher scripts for the driver. To run execute

    $ target/pack/bin/cooc
    
The driver will execute in Spark standalone mode and put the data in /path/to/3-input-cooc/data/indicators/*indicator-type*

##Using a Debugger
To build and run this example in a debugger like IntelliJ IDEA. Install from the IntelliJ site and add the Scala plugin.

Open IDEA and go to the menu File->New->Project from existing sources->SBT->/path/to/3-input-cooc. This will create an IDEA project from ```build.sbt``` in the root directory.

At this point you may create a "Debug Configuration" to run. In the menu choose Run->Edit Configurations. Under "Default" choose "Application". In the dialog hit the elipsis button "..." to the right of "Environment Variables" and fill in your versions of JAVA_HOME, SPARK_HOME, and MAHOUT_HOME. In configuration editor under "Use classpath from" choose root-3-input-cooc module. 

![image](http://mahout.apache.org/images/debug-config.png)

Now choose "Application" in the left pane and hit the plus sign "+". give the config a name and hit the elipsis button to the right of the "Main class" field as shown.

![image](http://mahout.apache.org/images/debug-config-2.png)


After setting breakpoints you are now ready to debug the configuration. Go to the Run->Debug... menu and pick your configuration. This will execute using a local standalone instance of Spark.

##The Mahout Shell

For small script-like apps you may wish to use the Mahout shell. It is a Scala REPL type interactive shell built on the Spark shell with Mahout-Samsara extensions.

To make the CooccurrenceDriver.scala into a script make the following changes:

* You won't need the context, since it is created when the shell is launched, comment that line out.
* Replace the logger.info lines with println
* Remove the package info since it's not needed, this will produce the file in ```path/to/3-input-cooc/bin/CooccurrenceDriver.mscala```. 

Note the extension ```.mscala``` to indicate we are using Mahout's scala extensions for math, otherwise known as [Mahout-Samsara](http://mahout.apache.org/users/environment/out-of-core-reference.html)

To run the code make sure the output does not exist already

    $ rm -r /path/to/3-input-cooc/data/indicators
    
Launch the Mahout + Spark shell:

    $ mahout spark-shell
    
You'll see the Mahout splash:

    MAHOUT_LOCAL is set, so we don't add HADOOP_CONF_DIR to classpath.

                         _                 _
             _ __ ___   __ _| |__   ___  _   _| |_
            | '_ ` _ \ / _` | '_ \ / _ \| | | | __|
            | | | | | | (_| | | | | (_) | |_| | |_
            |_| |_| |_|\__,_|_| |_|\___/ \__,_|\__|  version 0.10.0

      
    Using Scala version 2.10.4 (Java HotSpot(TM) 64-Bit Server VM, Java 1.7.0_72)
    Type in expressions to have them evaluated.
    Type :help for more information.
    15/04/26 09:30:48 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    Created spark context..
    Mahout distributed context is available as "implicit val sdc".
    mahout> 

To load the driver type:

    mahout> :load /path/to/3-input-cooc/bin/CooccurrenceDriver.mscala
    Loading ./bin/CooccurrenceDriver.mscala...
    import com.google.common.collect.{HashBiMap, BiMap}
    import org.apache.log4j.Logger
    import org.apache.mahout.math.cf.SimilarityAnalysis
    import org.apache.mahout.math.indexeddataset._
    import org.apache.mahout.sparkbindings._
    import scala.collection.immutable.HashMap
    defined module CooccurrenceDriver
    mahout> 

To run the driver type:

    mahout> CooccurrenceDriver.main(args = Array(""))
    
You'll get some stats printed:

    Total number of users for all actions = 5
    purchase indicator matrix:
      Number of rows for matrix = 4
      Number of columns for matrix = 5
      Number of rows after resize = 5
    view indicator matrix:
      Number of rows for matrix = 4
      Number of columns for matrix = 5
      Number of rows after resize = 5
    category indicator matrix:
      Number of rows for matrix = 5
      Number of columns for matrix = 7
      Number of rows after resize = 5
    
If you look in ```path/to/3-input-cooc/data/indicators``` you should find folders containing the indicator matrices.
