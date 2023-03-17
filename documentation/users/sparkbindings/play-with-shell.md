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
title: Perceptron and Winnow

    
---
# Playing with Mahout's Spark Shell

This tutorial will show you how to play with Mahout's scala DSL for linear algebra and its Spark shell. **Please keep in mind that this code is still in a very early experimental stage**.

_(Edited for 0.10.2)_

## Intro

We'll use an excerpt of a publicly available [dataset about cereals](http://lib.stat.cmu.edu/DASL/Datafiles/Cereals.html). The dataset tells the protein, fat, carbohydrate and sugars (in milligrams) contained in a set of cereals, as well as a customer rating for the cereals. Our aim for this example is to fit a linear model which infers the customer rating from the ingredients.


Name                    | protein | fat | carbo | sugars | rating
:-----------------------|:--------|:----|:------|:-------|:---------
Apple Cinnamon Cheerios | 2       | 2   | 10.5  | 10     | 29.509541
Cap'n'Crunch            | 1       | 2   | 12    | 12     | 18.042851  
Cocoa Puffs             | 1       | 1   | 12    | 13     | 22.736446
Froot Loops             | 2       |	1   | 11    | 13     | 32.207582  
Honey Graham Ohs        | 1       |	2   | 12    | 11     | 21.871292
Wheaties Honey Gold     | 2       | 1   | 16    |  8     | 36.187559  
Cheerios                | 6       |	2   | 17    |  1     | 50.764999
Clusters                | 3       |	2   | 13    |  7     | 40.400208
Great Grains Pecan      | 3       | 3   | 13    |  4     | 45.811716  


## Installing Mahout & Spark on your local machine

We describe how to do a quick toy setup of Spark & Mahout on your local machine, so that you can run this example and play with the shell. 

 1. Download [Apache Spark 1.6.2](http://d3kbcqa49mib13.cloudfront.net/spark-1.6.2-bin-hadoop2.6.tgz) and unpack the archive file
 1. Change to the directory where you unpacked Spark and type ```sbt/sbt assembly``` to build it
 1. Create a directory for Mahout somewhere on your machine, change to there and checkout the master branch of Apache Mahout from GitHub ```git clone https://github.com/apache/mahout mahout```
 1. Change to the ```mahout``` directory and build mahout using ```mvn -DskipTests clean install```
 
## Starting Mahout's Spark shell

 1. Goto the directory where you unpacked Spark and type ```sbin/start-all.sh``` to locally start Spark
 1. Open a browser, point it to [http://localhost:8080/](http://localhost:8080/) to check whether Spark successfully started. Copy the url of the spark master at the top of the page (it starts with **spark://**)
 1. Define the following environment variables: <pre class="codehilite">export MAHOUT_HOME=[directory into which you checked out Mahout]
export SPARK_HOME=[directory where you unpacked Spark]
export MASTER=[url of the Spark master]
</pre>
 1. Finally, change to the directory where you unpacked Mahout and type ```bin/mahout spark-shell```, 
you should see the shell starting and get the prompt ```mahout> ```. Check 
[FAQ](http://mahout.apache.org/users/sparkbindings/faq.html) for further troubleshooting.

## Implementation

We'll use the shell to interactively play with the data and incrementally implement a simple [linear regression](https://en.wikipedia.org/wiki/Linear_regression) algorithm. Let's first load the dataset. Usually, we wouldn't need Mahout unless we processed a large dataset stored in a distributed filesystem. But for the sake of this example, we'll use our tiny toy dataset and "pretend" it was too big to fit onto a single machine.

*Note: You can incrementally follow the example by copy-and-pasting the code into your running Mahout shell.*

Mahout's linear algebra DSL has an abstraction called *DistributedRowMatrix (DRM)* which models a matrix that is partitioned by rows and stored in the memory of a cluster of machines. We use ```dense()``` to create a dense in-memory matrix from our toy dataset and use ```drmParallelize``` to load it into the cluster, "mimicking" a large, partitioned dataset.

<div class="codehilite"><pre>
val drmData = drmParallelize(dense(
  (2, 2, 10.5, 10, 29.509541),  // Apple Cinnamon Cheerios
  (1, 2, 12,   12, 18.042851),  // Cap'n'Crunch
  (1, 1, 12,   13, 22.736446),  // Cocoa Puffs
  (2, 1, 11,   13, 32.207582),  // Froot Loops
  (1, 2, 12,   11, 21.871292),  // Honey Graham Ohs
  (2, 1, 16,   8,  36.187559),  // Wheaties Honey Gold
  (6, 2, 17,   1,  50.764999),  // Cheerios
  (3, 2, 13,   7,  40.400208),  // Clusters
  (3, 3, 13,   4,  45.811716)), // Great Grains Pecan
  numPartitions = 2);
</pre></div>

Have a look at this matrix. The first four columns represent the ingredients 
(our features) and the last column (the rating) is the target variable for 
our regression. [Linear regression](https://en.wikipedia.org/wiki/Linear_regression) 
assumes that the **target variable** `\(\mathbf{y}\)` is generated by the 
linear combination of **the feature matrix** `\(\mathbf{X}\)` with the 
**parameter vector** `\(\boldsymbol{\beta}\)` plus the
 **noise** `\(\boldsymbol{\varepsilon}\)`, summarized in the formula 
`\(\mathbf{y}=\mathbf{X}\boldsymbol{\beta}+\boldsymbol{\varepsilon}\)`. 
Our goal is to find an estimate of the parameter vector 
`\(\boldsymbol{\beta}\)` that explains the data very well.

As a first step, we extract `\(\mathbf{X}\)` and `\(\mathbf{y}\)` from our data matrix. We get *X* by slicing: we take all rows (denoted by ```::```) and the first four columns, which have the ingredients in milligrams as content. Note that the result is again a DRM. The shell will not execute this code yet, it saves the history of operations and defers the execution until we really access a result. **Mahout's DSL automatically optimizes and parallelizes all operations on DRMs and runs them on Apache Spark.**

<div class="codehilite"><pre>
val drmX = drmData(::, 0 until 4)
</pre></div>

Next, we extract the target variable vector *y*, the fifth column of the data matrix. We assume this one fits into our driver machine, so we fetch it into memory using ```collect```:

<div class="codehilite"><pre>
val y = drmData.collect(::, 4)
</pre></div>

Now we are ready to think about a mathematical way to estimate the parameter vector *β*. A simple textbook approach is [ordinary least squares (OLS)](https://en.wikipedia.org/wiki/Ordinary_least_squares), which minimizes the sum of residual squares between the true target variable and the prediction of the target variable. In OLS, there is even a closed form expression for estimating `\(\boldsymbol{\beta}\)` as 
`\(\left(\mathbf{X}^{\top}\mathbf{X}\right)^{-1}\mathbf{X}^{\top}\mathbf{y}\)`.

The first thing which we compute for this is  `\(\mathbf{X}^{\top}\mathbf{X}\)`. The code for doing this in Mahout's scala DSL maps directly to the mathematical formula. The operation ```.t()``` transposes a matrix and analogous to R ```%*%``` denotes matrix multiplication.

<div class="codehilite"><pre>
val drmXtX = drmX.t %*% drmX
</pre></div>

The same is true for computing `\(\mathbf{X}^{\top}\mathbf{y}\)`. We can simply type the math in scala expressions into the shell. Here, *X* lives in the cluster, while is *y* in the memory of the driver, and the result is a DRM again.
<div class="codehilite"><pre>
val drmXty = drmX.t %*% y
</pre></div>

We're nearly done. The next step we take is to fetch `\(\mathbf{X}^{\top}\mathbf{X}\)` and 
`\(\mathbf{X}^{\top}\mathbf{y}\)` into the memory of our driver machine (we are targeting 
features matrices that are tall and skinny , 
so we can assume that `\(\mathbf{X}^{\top}\mathbf{X}\)` is small enough 
to fit in). Then, we provide them to an in-memory solver (Mahout provides 
the an analog to R's ```solve()``` for that) which computes ```beta```, our 
OLS estimate of the parameter vector `\(\boldsymbol{\beta}\)`.

<div class="codehilite"><pre>
val XtX = drmXtX.collect
val Xty = drmXty.collect(::, 0)

val beta = solve(XtX, Xty)
</pre></div>

That's it! We have a implemented a distributed linear regression algorithm 
on Apache Spark. I hope you agree that we didn't have to worry a lot about 
parallelization and distributed systems. The goal of Mahout's linear algebra 
DSL is to abstract away the ugliness of programming a distributed system 
as much as possible, while still retaining decent performance and 
scalability.

We can now check how well our model fits its training data. 
First, we multiply the feature matrix `\(\mathbf{X}\)` by our estimate of 
`\(\boldsymbol{\beta}\)`. Then, we look at the difference (via L2-norm) of 
the target variable `\(\mathbf{y}\)` to the fitted target variable:

<div class="codehilite"><pre>
val yFitted = (drmX %*% beta).collect(::, 0)
(y - yFitted).norm(2)
</pre></div>

We hope that we could show that Mahout's shell allows people to interactively and incrementally write algorithms. We have entered a lot of individual commands, one-by-one, until we got the desired results. We can now refactor a little by wrapping our statements into easy-to-use functions. The definition of functions follows standard scala syntax. 

We put all the commands for ordinary least squares into a function ```ols```. 

<div class="codehilite"><pre>
def ols(drmX: DrmLike[Int], y: Vector) = 
  solve(drmX.t %*% drmX, drmX.t %*% y)(::, 0)

</pre></div>

Note that DSL declares implicit `collect` if coersion rules require an in-core argument. Hence, we can simply
skip explicit `collect`s. 

Next, we define a function ```goodnessOfFit``` that tells how well a model fits the target variable:

<div class="codehilite"><pre>
def goodnessOfFit(drmX: DrmLike[Int], beta: Vector, y: Vector) = {
  val fittedY = (drmX %*% beta).collect(::, 0)
  (y - fittedY).norm(2)
}
</pre></div>

So far we have left out an important aspect of a standard linear regression 
model. Usually there is a constant bias term added to the model. Without 
that, our model always crosses through the origin and we only learn the 
right angle. An easy way to add such a bias term to our model is to add a 
column of ones to the feature matrix `\(\mathbf{X}\)`. 
The corresponding weight in the parameter vector will then be the bias term.

Here is how we add a bias column:

<div class="codehilite"><pre>
val drmXwithBiasColumn = drmX cbind 1
</pre></div>

Now we can give the newly created DRM ```drmXwithBiasColumn``` to our model fitting method ```ols``` and see how well the resulting model fits the training data with ```goodnessOfFit```. You should see a large improvement in the result.

<div class="codehilite"><pre>
val betaWithBiasTerm = ols(drmXwithBiasColumn, y)
goodnessOfFit(drmXwithBiasColumn, betaWithBiasTerm, y)
</pre></div>

As a further optimization, we can make use of the DSL's caching functionality. We use ```drmXwithBiasColumn``` repeatedly  as input to a computation, so it might be beneficial to cache it in memory. This is achieved by calling ```checkpoint()```. In the end, we remove it from the cache with uncache:

<div class="codehilite"><pre>
val cachedDrmX = drmXwithBiasColumn.checkpoint()

val betaWithBiasTerm = ols(cachedDrmX, y)
val goodness = goodnessOfFit(cachedDrmX, betaWithBiasTerm, y)

cachedDrmX.uncache()

goodness
</pre></div>


Liked what you saw? Checkout Mahout's overview for the [Scala and Spark bindings](https://mahout.apache.org/users/sparkbindings/home.html).