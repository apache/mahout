/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.mahout.flinkbindings

import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.scala.DataSet
import org.apache.flink.api.scala.hadoop.mapreduce.HadoopOutputFormat
import org.apache.mahout.common.RandomUtils

import scala.collection.immutable.List

//import org.apache.flink.api.scala.hadoop.mapreduce.HadoopOutputFormat
import org.apache.hadoop.io.IntWritable
import org.apache.hadoop.mapreduce.Job
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, SequenceFileOutputFormat}
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import RLikeDrmOps._
import RLikeOps._
import math._

import org.apache.mahout.math.decompositions._
import org.scalatest.{FunSuite, Matchers}


import scala.reflect.ClassTag
import org.apache.flink.api.scala._



class FailingTestsSuite extends FunSuite with DistributedFlinkSuite with Matchers {

// // passing now
//  test("Simple DataSet to IntWritable") {
//    val path = TmpDir + "flinkOutput"
//
//    implicit val typeInfo = createTypeInformation[(Int,Int)]
//    val ds = env.fromElements[(Int,Int)]((1,2),(3,4),(5,6),(7,8))
//   // val job = new JobConf
//
//
//    val writableDataset : DataSet[(IntWritable,IntWritable)] =
//      ds.map( tuple =>
//        (new IntWritable(tuple._1.asInstanceOf[Int]), new IntWritable(tuple._2.asInstanceOf[Int]))
//    )
//
//    val job: Job = new Job()
//
//    job.setOutputKeyClass(classOf[IntWritable])
//    job.setOutputValueClass(classOf[IntWritable])
//
//    // setup sink for IntWritable
//    val sequenceFormat = new SequenceFileOutputFormat[IntWritable, IntWritable]
//    val hadoopOutput  = new HadoopOutputFormat[IntWritable,IntWritable](sequenceFormat, job)
//    FileOutputFormat.setOutputPath(job, new org.apache.hadoop.fs.Path(path))
//
//    writableDataset.output(hadoopOutput)
//
//    env.execute(s"dfsWrite($path)")
//
//  }


//  test("C = A + B, identically partitioned") {
//
//    val inCoreA = dense((1, 2, 3), (3, 4, 5), (5, 6, 7))
//
//    val A = drmParallelize(inCoreA, numPartitions = 2)
//
//     //   printf("A.nrow=%d.\n", A.rdd.count())
//
//    // Create B which would be identically partitioned to A. mapBlock() by default will do the trick.
//    val B = A.mapBlock() {
//      case (keys, block) =>
//        val bBlock = block.like() := { (r, c, v) => util.Random.nextDouble()}
//        keys -> bBlock
//    }
//      // Prevent repeated computation non-determinism
//      // flink problem is here... checkpoint is not doing what it should
//      // ie. greate a physical plan w/o side effects
//      .checkpoint()
//
//    val inCoreB = B.collect
//
//    printf("A=\n%s\n", inCoreA)
//    printf("B=\n%s\n", inCoreB)
//
//    val C = A + B
//
//    val inCoreC = C.collect
//
//    printf("C=\n%s\n", inCoreC)
//
//    // Actual
//    val inCoreCControl = inCoreA + inCoreB
//
//    (inCoreC - inCoreCControl).norm should be < 1E-10
//  }
//// Passing now.
//  test("C = inCoreA %*%: B") {
//
//    val inCoreA = dense((1, 2, 3), (3, 4, 5), (4, 5, 6), (5, 6, 7))
//    val inCoreB = dense((3, 5, 7, 10), (4, 6, 9, 10), (5, 6, 7, 7))
//
//    val B = drmParallelize(inCoreB, numPartitions = 2)
//    val C = inCoreA %*%: B
//
//    val inCoreC = C.collect
//    val inCoreCControl = inCoreA %*% inCoreB
//
//    println(inCoreC)
//    (inCoreC - inCoreCControl).norm should be < 1E-10
//
//  }
//
//  test("dsqDist(X,Y)") {
//    val m = 100
//    val n = 300
//    val d = 7
//    val mxX = Matrices.symmetricUniformView(m, d, 12345).cloned -= 5
//    val mxY = Matrices.symmetricUniformView(n, d, 1234).cloned += 10
//    val (drmX, drmY) = (drmParallelize(mxX, 3), drmParallelize(mxY, 4))
//
//    val mxDsq = dsqDist(drmX, drmY).collect
//    val mxDsqControl = new DenseMatrix(m, n) := { (r, c, _) ⇒ (mxX(r, ::) - mxY(c, ::)) ^= 2 sum }
//    (mxDsq - mxDsqControl).norm should be < 1e-7
//  }
//
//  test("dsqDist(X)") {
//    val m = 100
//    val d = 7
//    val mxX = Matrices.symmetricUniformView(m, d, 12345).cloned -= 5
//    val drmX = drmParallelize(mxX, 3)
//
//    val mxDsq = dsqDist(drmX).collect
//    val mxDsqControl = sqDist(drmX)
//    (mxDsq - mxDsqControl).norm should be < 1e-7
//  }

//// passing now
//  test("DRM DFS i/o (local)") {
//
//    val uploadPath = TmpDir + "UploadedDRM"
//
//    val inCoreA = dense((1, 2, 3), (3, 4, 5))
//    val drmA = drmParallelize(inCoreA)
//
//    drmA.dfsWrite(path = uploadPath)
//
//    println(inCoreA)
//
//    // Load back from hdfs
//    val drmB = drmDfsRead(path = uploadPath)
//
//    // Make sure keys are correctly identified as ints
//    drmB.checkpoint(CacheHint.NONE).keyClassTag shouldBe ClassTag.Int
//
//    // Collect back into in-core
//    val inCoreB = drmB.collect
//
//    // Print out to see what it is we collected:
//    println(inCoreB)
//
//    (inCoreA - inCoreB).norm should be < 1e-7
//  }



//  test("dspca") {
//
//    val rnd = RandomUtils.getRandom
//
//    // Number of points
//    val m = 500
//    // Length of actual spectrum
//    val spectrumLen = 40
//
//    val spectrum = dvec((0 until spectrumLen).map(x => 300.0 * exp(-x) max 1e-3))
//    printf("spectrum:%s\n", spectrum)
//
//    val (u, _) = qr(new SparseRowMatrix(m, spectrumLen) :=
//      ((r, c, v) => if (rnd.nextDouble() < 0.2) 0 else rnd.nextDouble() + 5.0))
//
//    // PCA Rotation matrix -- should also be orthonormal.
//    val (tr, _) = qr(Matrices.symmetricUniformView(spectrumLen, spectrumLen, rnd.nextInt) - 10.0)
//
//    val input = (u %*%: diagv(spectrum)) %*% tr.t
//    val drmInput = drmParallelize(m = input, numPartitions = 2)
//
//    // Calculate just first 10 principal factors and reduce dimensionality.
//    // Since we assert just validity of the s-pca, not stochastic error, we bump p parameter to
//    // ensure to zero stochastic error and assert only functional correctness of the method's pca-
//    // specific additions.
//    val k = 10
//
//    // Calculate just first 10 principal factors and reduce dimensionality.
//    var (drmPCA, _, s) = dspca(drmA = drmInput, k = 10, p = spectrumLen, q = 1)
//    // Un-normalized pca data:
//    drmPCA = drmPCA %*% diagv(s)
//
//    val pca = drmPCA.checkpoint(CacheHint.NONE).collect
//
//    // Of course, once we calculated the pca, the spectrum is going to be different since our originally
//    // generated input was not centered. So here, we'd just brute-solve pca to verify
//    val xi = input.colMeans()
//    for (r <- 0 until input.nrow) input(r, ::) -= xi
//    var (pcaControl, _, sControl) = svd(m = input)
//    pcaControl = (pcaControl %*%: diagv(sControl))(::, 0 until k)
//
//    printf("pca:\n%s\n", pca(0 until 10, 0 until 10))
//    printf("pcaControl:\n%s\n", pcaControl(0 until 10, 0 until 10))
//
//    (pca(0 until 10, 0 until 10).norm - pcaControl(0 until 10, 0 until 10).norm).abs should be < 1E-5
//
//  }

  test("dals") {

    val rnd = RandomUtils.getRandom

    // Number of points
    val m = 500
    val n = 500

    // Length of actual spectrum
    val spectrumLen = 40

    // Create singluar values with decay
    val spectrum = dvec((0 until spectrumLen).map(x => 300.0 * exp(-x) max 1e-3))
    printf("spectrum:%s\n", spectrum)

    // Create A as an ideal input
    val inCoreA = (qr(Matrices.symmetricUniformView(m, spectrumLen, 1234))._1 %*%: diagv(spectrum)) %*%
      qr(Matrices.symmetricUniformView(n, spectrumLen, 2345))._1.t
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    // Decompose using ALS
    val (drmU, drmV, rmse) = dals(drmA = drmA, k = 20).toTuple
    val inCoreU = drmU.collect
    val inCoreV = drmV.collect

    val predict = inCoreU %*% inCoreV.t

    printf("Control block:\n%s\n", inCoreA(0 until 3, 0 until 3))
    printf("ALS factorized approximation block:\n%s\n", predict(0 until 3, 0 until 3))

    val err = (inCoreA - predict).norm
    printf("norm of residuals %f\n", err)
    printf("train iteration rmses: %s\n", rmse)

    err should be < 15e-2

  }

}