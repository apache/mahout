Welcome to Apache Mahout!
===========
The Apache Mahout™ project's goal is to build an environment for quickly creating scalable performant machine learning applications using Scala + Spark (H2O in progress) to mature Hadoop's MapReduce algorithms.

For additional information about Mahout, checkout the [Mahout Home Page](http://mahout.apache.org/)
####Installing Mahout
To compile from source:
* `mvn -DskipTests clean install`
* To run tests do `mvn test`
* To set up your IDE, do `mvn eclipse:eclipse` or `mvn idea:idea`

To use Maven:
```
<dependency>
    <groupId>org.apache.mahout</groupId>
    <artifactId>mahout-mr</artifactId>
    <version>0.10.0</version>
</dependency>
```

####Examples
For examples of how to properly use Mahout, see the examples directory located in `examples/bin`


####For information on how to contribute see:
  https://mahout.apache.org/developers/how-to-contribute.html

Legal
 Please see the `NOTICE.txt` included in this directory for more information.
