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
title: (Deprecated)  Class Discovery

    
---
<a name="ClassDiscovery-ClassDiscovery"></a>
# Class Discovery

See http://www.cs.bham.ac.uk/~wbl/biblio/gecco1999/GP-417.pdf

CDGA uses a Genetic Algorithm to discover a classification rule for a given
dataset. 
A dataset can be seen as a table:

<table>
<tr><th> </th><th>attribute 1</th><th>attribute 2</th><th>...</th><th>attribute N</th></tr>
<tr><td>row 1</td><td>value1</td><td>value2</td><td>...</td><td>valueN</td></tr>
<tr><td>row 2</td><td>value1</td><td>value2</td><td>...</td><td>valueN</td></tr>
<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
<tr><td>row M</td><td>value1</td><td>value2</td><td>...</td><td>valueN</td></tr>
</table>

An attribute can be numerical, for example a "temperature" attribute, or
categorical, for example a "color" attribute. For classification purposes,
one of the categorical attributes is designated as a *label*, which means
that its value defines the *class* of the rows.
A classification rule can be represented as follows:
<table>
<tr><th> </th><th>attribute 1</th><th>attribute 2</th><th>...</th><th>attribute N</th></tr>
<tr><td>weight</td><td>w1</td><td>w2</td><td>...</td><td>wN</td></tr>
<tr><td>operator</td><td>op1</td><td>op2</td><td>...</td><td>opN</td></tr>
<tr><td>value</td><td>value1</td><td>value2</td><td>...</td><td>valueN</td></tr>
</table>

For a given *target* class and a weight *threshold*, the classification
rule can be read :


    for each row of the dataset
      if (rule.w1 < threshold || (rule.w1 >= threshold && row.value1 rule.op1
rule.value1)) &&
         (rule.w2 < threshold || (rule.w2 >= threshold && row.value2 rule.op2
rule.value2)) &&
         ...
         (rule.wN < threshold || (rule.wN >= threshold && row.valueN rule.opN
rule.valueN)) then
        row is part of the target class


*Important:* The label attribute is not evaluated by the rule.

The threshold parameter allows some conditions of the rule to be skipped if
their weight is too small. The operators available depend on the attribute
types:
* for a numerical attributes, the available operators are '<' and '>='
* for categorical attributes, the available operators are '!=' and '=='

The "threshold" and "target" are user defined parameters, and because the
label is always a categorical attribute, the target is the (zero based)
index of the class label value in all the possible values of the label. For
example, if the label attribute can have the following values (blue, brown,
green), then a target of 1 means the "blue" class.

For example, we have the following dataset (the label attribute is "Eyes
Color"):
<table>
<tr><th> </th><th>Age</th><th>Eyes Color</th><th>Hair Color</th></tr>
<tr><td>row 1</td><td>16</td><td>brown</td><td>dark</td></tr>
<tr><td>row 2</td><td>25</td><td>green</td><td>light</td></tr>
<tr><td>row 3</td><td>12</td><td>blue</td><td>light</td></tr>
and a classification rule:
<tr><td>weight</td><td>0</td><td>1</td></tr>
<tr><td>operator</td><td><</td><td>!=</td></tr>
<tr><td>value</td><td>20</td><td>light</td></tr>
and the following parameters: threshold = 1 and target = 0 (brown).
</table>

This rule can be read as follows:

    for each row of the dataset
      if (0 < 1 || (0 >= 1 && row.value1 < 20)) &&
         (1 < 1 || (1 >= 1 && row.value2 != light)) then
        row is part of the "brown Eye Color" class


Please note how the rule skipped the label attribute (Eye Color), and how
the first condition is ignored because its weight is < threshold.

<a name="ClassDiscovery-Runningtheexample:"></a>
# Running the example:
NOTE: Substitute in the appropriate version for the Mahout JOB jar

1. cd <MAHOUT_HOME>/examples
1. ant job
1. {code}<HADOOP_HOME>/bin/hadoop dfs -put
<MAHOUT_HOME>/examples/src/test/resources/wdbc wdbc{code}
1. {code}<HADOOP_HOME>/bin/hadoop dfs -put
<MAHOUT_HOME>/examples/src/test/resources/wdbc.infos wdbc.infos{code}
1. {code}<HADOOP_HOME>/bin/hadoop jar
<MAHOUT_HOME>/examples/build/apache-mahout-examples-0.1-dev.job
org.apache.mahout.ga.watchmaker.cd.CDGA
<MAHOUT_HOME>/examples/src/test/resources/wdbc 1 0.9 1 0.033 0.1 0 100 10

    CDGA needs 9 parameters:
    * param 1 : path of the directory that contains the dataset and its infos
file
    * param 2 : target class
    * param 3 : threshold
    * param 4 : number of crossover points for the multi-point crossover
    * param 5 : mutation rate
    * param 6 : mutation range
    * param 7 : mutation precision
    * param 8 : population size
    * param 9 : number of generations before the program stops
    
    For more information about 4th parameter, please see [Multi-point Crossover|http://www.geatbx.com/docu/algindex-03.html#P616_36571]
.
    For a detailed explanation about the 5th, 6th and 7th parameters, please
see [Real Valued Mutation|http://www.geatbx.com/docu/algindex-04.html#P659_42386]
.
    
    *TODO*: Fill in where to find the output and what it means.
    
    h1. The info file:
    To run properly, CDGA needs some informations about the dataset. Each
dataset should be accompanied by an .infos file that contains the needed
informations. for each attribute a corresponding line in the info file
describes it, it can be one of the following:
    * IGNORED
      if the attribute is ignored
    * LABEL, val1, val2,...
      if the attribute is the label (class), and its possible values
    * CATEGORICAL, val1, val2,...
      if the attribute is categorial (nominal), and its possible values
    * NUMERICAL, min, max
      if the attribute is numerical, and its min and max values
    
    This file can be generated automaticaly using a special tool available with
CDGA.
    


*  the tool searches for an existing infos file (*must be filled by the
user*), in the same directory of the dataset with the same name and with
the ".infos" extension, that contain the type of the attributes:
  ** 'N' numerical attribute
  ** 'C' categorical attribute
  ** 'L' label (this also a categorical attribute)
  ** 'I' to ignore the attribute
  each attribute is in a separate 
* A Hadoop job is used to parse the dataset and collect the informations.
This means that *the dataset can be distributed over HDFS*.
* the results are written back in the same .info file, with the correct
format needed by CDGA.
