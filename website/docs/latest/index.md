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
title: Welcome to the Docs
tagline: Apache Mahout from 30,000 feet (10,000 meters)
---


You've probably already noticed Mahout has a lot of things going on at different levels, and it can be hard to know where
to start.  Let's provide an overview to help you see how the pieces fit together. In general the stack is something like this:

1. Application Code
1. Samsara Scala-DSL (Syntactic Sugar)
1. Logical/Physical DAG
1. Engine Bindings
1. Code runs in Engine
1. Native Solvers 

## Application Code

You have an JAVA/Scala applicatoin (skip this if you're working from an interactive shell or Apache Zeppelin)

    
    def main(args: Array[String]) {

      println("Welcome to My Mahout App")

      if (args.isEmpty) {

This may seem like a trivial part to call out, but the point is important- Mahout runs _inline_ with your regular application 
code. E.g. if this is an Apache Spark app, then you do all your Spark things, including ETL and data prep in the same 
application, and then invoke Mahout's mathematically expressive Scala DSL when you're ready to math on it.

## Samsara Scala-DSL (Syntactic Sugar)

So when you get to a point in your code where you're ready to math it up (in this example Spark) you can elegantly express 
yourself mathematically.

    implicit val sdc: org.apache.mahout.sparkbindings.SparkDistributedContext = sc2sdc(sc)
    
    val A = drmWrap(rddA)
    val B = drmWrap(rddB) 
    
    val C = A.t %*% A + A %*% B.t
    
We've defined a `MahoutDistributedContext` (which is a wrapper on the Spark Context), and two Distributed Row Matrices (DRMs)
which are wrappers around RDDs (in Spark).  

## Logical / Physical DAG

At this point there is a bit of optimization that happens.  For example, consider the
    
    A.t %*% A
    
Which is 
<center>\(\mathbf{A^\intercal A}\)</center>

Transposing a large matrix is a very expensive thing to do, and in this case we don't actually need to do it: there is a
more efficient way to calculate <foo>\(\mathbf{A^\intercal A}\)</foo> that doesn't require a physical transpose. 

(Image showing this)

Mahout converts this code into something that looks like:

    OpAtA(A) + OpABt(A, B) //  illustrative pseudocode with real functions called

There's a little more magic that happens at this level, but the punchline is _Mahout translates the pretty scala into a
a series of operators, which are implemented at engine level_.

## Engine Bindings and Engine Level Ops

When one creates new engine bindings, one is in essence defining:

1. What the engine specific underlying structure for a DRM is (in Spark its an RDD).  The underlying structure also has 
rows of `MahoutVector`s, so in Spark `RDD[(index, MahoutVector)]`.  This will be important when we get to the native solvers. 
1. Implementing a set of BLAS (basic linear algebra) functions for working on the underlying structure- in Spark this means
implementing things like `AtA` on an RDD. See [the sparkbindings on github](https://github.com/apache/mahout/tree/master/spark/src/main/scala/org/apache/mahout/sparkbindings)

Now your mathematically expressive Samsara Scala code has been translated into optimized engine specific functions.

## Native Solvers

Recall how I said that rows of the DRMs are `org.apache.mahout.math.Vector`.  Here is where this becomes important. I'm going 
to explain this in the context of Spark, but the principals apply to all distributed backends. 

If you are familiar with how mapping and reducing in Spark, then envision this RDD of `MahoutVector`s,  each partition, 
and indexed collection of vectors is a _block_ of the distributed matrix, however this _block_ is totally in-core, and therefor is treated like an in-core matrix. 

Now Mahout defines its own in-core BLAS packs and refers to them as _Native Solvers_.  The default native solver is just plain
old JVM, which is painfully slow, but works just about anywhere.  

When the data gets to the node, an operation on the matrix block is called.  In the same way Mahout converts abstract
operators on the DRM that are implemented on various distributed engines, it calls abstract operators on the in-core matrix 
and vectors which are implemented on various native solvers. 

The default "native solver" is the JVM, which isn't native at all, and if no actual native solvers are present operations 
will fall back to this. However, IF a native solver is present (the jar was added to the notebook), then the magic will happen.

Imagine still we have our Spark executor: it has this block of a matrix sitting in its core. Now let's suppose the `ViennaCl-OMP`
native solver is in use.  When Spark calls an operation on this incore matrix, the matrix dumps out of the JVM and the 
calculation is carried out on _all available CPUs_. 

In a similar way, the `ViennaCL` native solver dumps the matrix out of the JVM and looks for a GPU to execute the operations on.
 
Once the operations are complete, the result is loaded back up into the JVM, and Spark (or whatever distributed engine) and 
shipped back to the driver. 

The native solver operations are only defined on `org.apache.mahout.math.Vector` and `org.apache.mahout.math.Matrix`, which is 
why it is critical that the underlying structure is composed row-wise by `Vector` or `Matrices`. 

