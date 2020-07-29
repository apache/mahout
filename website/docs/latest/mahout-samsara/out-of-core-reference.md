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
title: Mahout Samsara Out of Core

    
---
# Mahout-Samsara's Distributed Linear Algebra DSL Reference

**Note: this page is meant only as a quick reference to Mahout-Samsara's R-Like DSL semantics.  For more information, including information on Mahout-Samsara's Algebraic Optimizer please see: [Mahout Scala Bindings and Mahout Spark Bindings for Linear Algebra Subroutines](http://mahout.apache.org/users/sparkbindings/ScalaSparkBindings.pdf).**

The subjects of this reference are solely applicable to Mahout-Samsara's **DRM** (distributed row matrix).

In this reference, DRMs will be denoted as e.g. `A`, and in-core matrices as e.g. `inCoreA`.

#### Imports 

The following imports are used to enable seamless in-core and distributed algebraic DSL operations:

    import org.apache.mahout.math._
    import scalabindings._
    import RLikeOps._
    import drm._
    import RLikeDRMOps._
    
If working with mixed scala/java code:
    
    import collection._
    import JavaConversions._
    
If you are working with Mahout-Samsara's Spark-specific operations e.g. for context creation:

    import org.apache.mahout.sparkbindings._
    
The Mahout shell does all of these imports automatically.


#### DRM Persistence operators

**Mahout-Samsara's DRM persistance to HDFS is compatible with all Mahout-MapReduce algorithms such as seq2sparse.**


Loading a DRM from (HD)FS:

    drmDfsRead(path = hdfsPath)
     
Parallelizing from an in-core matrix:

    val inCoreA = (dense(1, 2, 3), (3, 4, 5))
    val A = drmParallelize(inCoreA)
    
Creating an empty DRM:

    val A = drmParallelizeEmpty(100, 50)
    
Collecting to driver's jvm in-core:

    val inCoreA = A.collect
    
**Warning: The collection of distributed matrices happens implicitly whenever conversion to an in-core (o.a.m.math.Matrix) type is required. E.g.:**

    val inCoreA: Matrix = ...
    val drmB: DrmLike[Int] =...
    val inCoreC: Matrix = inCoreA %*%: drmB
    
**implies (incoreA %*%: drmB).collect**

Collecting to (HD)FS as a Mahout's DRM formatted file:

    A.dfsWrite(path = hdfsPath)
    
#### Logical algebraic operators on DRM matrices:

A logical set of operators are defined for distributed matrices as a subset of those defined for in-core matrices.  In particular, since all distributed matrices are immutable, there are no assignment operators (e.g. **A += B**)
*Note: please see: [Mahout Scala Bindings and Mahout Spark Bindings for Linear Algebra Subroutines](http://mahout.apache.org/users/sparkbindings/ScalaSparkBindings.pdf) for information on Mahout-Samsars's Algebraic Optimizer, and translation from logical operations to a physical plan for the back end.*
 
    
Cache a DRM and trigger an optimized physical plan: 

    drmA.checkpoint(CacheHint.MEMORY_AND_DISK)
   
Other valid caching Instructions:

    drmA.checkpoint(CacheHint.NONE)
    drmA.checkpoint(CacheHint.DISK_ONLY)
    drmA.checkpoint(CacheHint.DISK_ONLY_2)
    drmA.checkpoint(CacheHint.MEMORY_ONLY)
    drmA.checkpoint(CacheHint.MEMORY_ONLY_2)
    drmA.checkpoint(CacheHint.MEMORY_ONLY_SER
    drmA.checkpoint(CacheHint.MEMORY_ONLY_SER_2)
    drmA.checkpoint(CacheHint.MEMORY_AND_DISK_2)
    drmA.checkpoint(CacheHint.MEMORY_AND_DISK_SER)
    drmA.checkpoint(CacheHint.MEMORY_AND_DISK_SER_2)

*Note: Logical DRM operations are lazily computed.  Currently the actual computations and optional caching will be triggered by dfsWrite(...), collect(...) and blockify(...).*



Transposition:

    A.t
 
Elementwise addition *(Matrices of identical geometry and row key types)*:
  
    A + B

Elementwise subtraction *(Matrices of identical geometry and row key types)*:

    A - B
    
Elementwise multiplication (Hadamard) *(Matrices of identical geometry and row key types)*:

    A * B
    
Elementwise division *(Matrices of identical geometry and row key types)*:

    A / B
    
**Elementwise operations involving one in-core argument (int-keyed DRMs only)**:

    A + inCoreB
    A - inCoreB
    A * inCoreB
    A / inCoreB
    A :+ inCoreB
    A :- inCoreB
    A :* inCoreB
    A :/ inCoreB
    inCoreA +: B
    inCoreA -: B
    inCoreA *: B
    inCoreA /: B

Note the Spark associativity change (e.g. `A *: inCoreB` means `B.leftMultiply(A`), same as when both arguments are in core). Whenever operator arguments include both in-core and out-of-core arguments, the operator can only be associated with the out-of-core (DRM) argument to support the distributed implementation.
    
**Matrix-matrix multiplication %*%**:

`\(\mathbf{M}=\mathbf{AB}\)`

    A %*% B
    A %*% inCoreB
    A %*% inCoreDiagonal
    A %*%: B


*Note: same as above, whenever operator arguments include both in-core and out-of-core arguments, the operator can only be associated with the out-of-core (DRM) argument to support the distributed implementation.*
 
**Matrix-vector multiplication %*%**
Currently we support a right multiply product of a DRM and an in-core Vector(`\(\mathbf{Ax}\)`) resulting in a single column DRM, which then can be collected in front (usually the desired outcome):

    val Ax = A %*% x
    val inCoreX = Ax.collect(::, 0)
    

**Matrix-scalar +,-,*,/**
Elementwise operations of every matrix element and a scalar:

    A + 5.0
    A - 5.0
    A :- 5.0
    5.0 -: A
    A * 5.0
    A / 5.0
    5.0 /: a
    
Note that `5.0 -: A` means `\(m_{ij} = 5 - a_{ij}\)` and `5.0 /: A` means `\(m_{ij} = \frac{5}{a{ij}}\)` for all elements of the result.
    
    
#### Slicing

General slice:

    A(100 to 200, 100 to 200)
    
Horizontal Block:

    A(::, 100 to 200)
    
Vertical Block:

    A(100 to 200, ::)
    
*Note: if row range is not all-range (::) the the DRM must be `Int`-keyed.  General case row slicing is not supported by DRMs with key types other than `Int`*.


#### Stitching

Stitch side by side (cbind R semantics):

    val drmAnextToB = drmA cbind drmB
    
Stitch side by side (Scala):

    val drmAnextToB = drmA.cbind(drmB)
    
Analogously, vertical concatenation is available via **rbind**

#### Custom pipelines on blocks
Internally, Mahout-Samsara's DRM is represented as a distributed set of vertical (Key, Block) tuples.

**drm.mapBlock(...)**:

The DRM operator `mapBlock` provides transformational access to the distributed vertical blockified tuples of a matrix (Row-Keys, Vertical-Matrix-Block).

Using `mapBlock` to add 1.0 to a DRM:

    val inCoreA = dense((1, 2, 3), (2, 3 , 4), (3, 4, 5))
    val drmA = drmParallelize(inCoreA)
    val B = A.mapBlock() {
        case (keys, block) => keys -> (block += 1.0)
    }
    
#### Broadcasting Vectors and matrices to closures
Generally we can create and use one-way closure attributes to be used on the back end.

Scalar matrix multiplication:

    val factor: Int = 15
    val drm2 = drm1.mapBlock() {
        case (keys, block) => block *= factor
        keys -> block
    }

**Closure attributes must be java-serializable. Currently Mahout's in-core Vectors and Matrices are not java-serializable, and must be broadcast to the closure using `drmBroadcast(...)`**:

    val v: Vector ...
    val bcastV = drmBroadcast(v)
    val drm2 = drm1.mapBlock() {
        case (keys, block) =>
            for(row <- 0 until block.nrow) block(row, ::) -= bcastV
        keys -> block    
    }

#### Computations providing ad-hoc summaries


Matrix cardinality:

    drmA.nrow
    drmA.ncol

*Note: depending on the stage of optimization, these may trigger a computational action.  I.e. if one calls `nrow()` n times, then the back end will actually recompute `nrow` n times.*
    
Means and sums:

    drmA.colSums
    drmA.colMeans
    drmA.rowSums
    drmA.rowMeans
    
 
*Note: These will always trigger a computational action.  I.e. if one calls `colSums()` n times, then the back end will actually recompute `colSums` n times.*

#### Distributed Matrix Decompositions

To import the decomposition package:
    
    import org.apache.mahout.math._
    import decompositions._
    
Distributed thin QR:

    val (drmQ, incoreR) = dqrThin(drmA)
    
Distributed SSVD:
 
    val (drmU, drmV, s) = dssvd(drmA, k = 40, q = 1)
    
Distributed SPCA:

    val (drmU, drmV, s) = dspca(drmA, k = 30, q = 1)

Distributed regularized ALS:

    val (drmU, drmV, i) = dals(drmA,
                            k = 50,
                            lambda = 0.0,
                            maxIterations = 10,
                            convergenceThreshold = 0.10))
                            
#### Adjusting parallelism of computations

Set the minimum parallelism to 100 for computations on `drmA`:

    drmA.par(min = 100)
 
Set the exact parallelism to 100 for computations on `drmA`:

    drmA.par(exact = 100)


Set the engine specific automatic parallelism adjustment for computations on `drmA`:

    drmA.par(auto = true)

#### Retrieving the engine specific data structure backing the DRM:

**A Spark RDD:**

    val myRDD = drmA.checkpoint().rdd
    
**An H2O Frame and Key Vec:**

    val myFrame = drmA.frame
    val myKeys = drmA.keys
    
**A Flink DataSet:**

    val myDataSet = drmA.ds
    
For more information including information on Mahout-Samsara's Algebraic Optimizer and in-core Linear algebra bindings see: [Mahout Scala Bindings and Mahout Spark Bindings for Linear Algebra Subroutines](http://mahout.apache.org/users/sparkbindings/ScalaSparkBindings.pdf)



    



