/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf

import org.scalatest.FunSuite
import org.apache.mahout.test.MahoutSuite
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.MatrixOps
import org.apache.mahout.math.drm._
import org.apache.mahout.math._
import org.apache.mahout.sparkbindings.test.MahoutLocalContext

/* values 
A =
1	1	0	0	0
0	0	1	1	0
0	0	0	0	1
1	0	0	1	0

B =
1	1	1	1	0
1	1	1	1	0
0	0	1	0	1
1	1	0	1	0
 */

class CooccurrenceAnalysisSuite extends FunSuite with MahoutSuite with MahoutLocalContext {

  test("cooccurrence [A'A], [B'A] boolbean data using LLR") {
    val a = dense((1, 1, 0, 0, 0), (0, 0, 1, 1, 0), (0, 0, 0, 0, 1), (1, 0, 0, 1, 0))
    val b = dense((1, 1, 1, 1, 0), (1, 1, 1, 1, 0), (0, 0, 1, 0, 1), (1, 1, 0, 1, 0))
    val drmA = drmParallelize(m = a, numPartitions = 2)
    val drmB = drmParallelize(m = b, numPartitions = 2)

    // correct cooccurrence with LLR
    val matrixLLRCoocAtAControl = dense(
      (0.0, 1.7260924347106847, 0, 0, 0),
      (1.7260924347106847, 0, 0, 0, 0),
      (0, 0, 0, 1.7260924347106847, 0),
      (0, 0, 1.7260924347106847, 0, 0),
      (0, 0, 0, 0, 0)
    )

    // correct cross-cooccurrence with LLR
    val matrixLLRCoocBtAControl = dense(
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0.6795961471815897),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (0, 0, 0, 0, 4.498681156950466)
    )

    //self similarity
    val drmCooc = CooccurrenceAnalysis.cooccurrences(drmARaw = drmA, drmBs = Array(drmB))
    val matrixSelfCooc = drmCooc(0).checkpoint().collect
    val diffMatrix = matrixSelfCooc.minus(matrixLLRCoocAtAControl)
    var n = (new MatrixOps(m = diffMatrix)).norm
    n should be < 1E-10

    //cross similarity
    val matrixCrossCooc = drmCooc(1).checkpoint().collect
    val diff2Matrix = matrixCrossCooc.minus(matrixLLRCoocBtAControl)
    n = (new MatrixOps(m = diff2Matrix)).norm
    n should be < 1E-10
  }

  test("cooccurrence [A'A], [B'A] double data using LLR") {
    val a = dense((100000.0D, 1.0D, 0.0D, 0.0D, 0.0D), (0.0D, 0.0D, 10.0D, 1.0D, 0.0D), (0.0D, 0.0D, 0.0D, 0.0D, 1000.0D), (1.0D, 0.0D, 0.0D, 10.0D, 0.0D))
    val b = dense((10000.0D, 100.0D, 1000.0D, 1.0D, 0.0D), (10.0D, 1.0D, 10000000.0D, 10.0D, 0.0D), (0.0D, 0.0D, 1000.0D, 0.0D, 100.0D), (100.0D, 1.0D, 0.0D, 100000.0D, 0.0D))
    val drmA = drmParallelize(m = a, numPartitions = 2)
    val drmB = drmParallelize(m = b, numPartitions = 2)

    // correct cooccurrence with LLR
    val matrixLLRCoocAtAControl = dense(
      (0.0, 1.7260924347106847, 0, 0, 0),
      (1.7260924347106847, 0, 0, 0, 0),
      (0, 0, 0, 1.7260924347106847, 0),
      (0, 0, 1.7260924347106847, 0, 0),
      (0, 0, 0, 0, 0)
    )

    // correct cross-cooccurrence with LLR
    val matrixLLRCoocBtAControl = dense(
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0.6795961471815897),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (0, 0, 0, 0, 4.498681156950466)
    )

    //self similarity
    val drmCooc = CooccurrenceAnalysis.cooccurrences(drmARaw = drmA, drmBs = Array(drmB))
    val matrixSelfCooc = drmCooc(0).checkpoint().collect
    val diffMatrix = matrixSelfCooc.minus(matrixLLRCoocAtAControl)
    var n = (new MatrixOps(m = diffMatrix)).norm
    n should be < 1E-10

    //cross similarity
    val matrixCrossCooc = drmCooc(1).checkpoint().collect
    val diff2Matrix = matrixCrossCooc.minus(matrixLLRCoocBtAControl)
    n = (new MatrixOps(m = diff2Matrix)).norm
    n should be < 1E-10
  }

  test("cooccurrence [A'A], [B'A] integer data using LLR") {
    val a = dense((1000, 10, 0, 0, 0), (0, 0, -10000, 10, 0), (0, 0, 0, 0, 100), (10000, 0, 0, 1000, 0))
    val b = dense((100, 1000, -10000, 10000, 0), (10000, 1000, 100, 10, 0), (0, 0, 10, 0, -100), (10, 100, 0, 1000, 0))
    val drmA = drmParallelize(m = a, numPartitions = 2)
    val drmB = drmParallelize(m = b, numPartitions = 2)

    // correct cooccurrence with LLR
    val matrixLLRCoocAtAControl = dense(
      (0.0, 1.7260924347106847, 0, 0, 0),
      (1.7260924347106847, 0, 0, 0, 0),
      (0, 0, 0, 1.7260924347106847, 0),
      (0, 0, 1.7260924347106847, 0, 0),
      (0, 0, 0, 0, 0)
    )

    // correct cross-cooccurrence with LLR
    val matrixLLRCoocBtAControl = dense(
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0.6795961471815897),
      (1.7260924347106847, 0.6795961471815897, 0.6795961471815897, 1.7260924347106847, 0),
      (0, 0, 0, 0, 4.498681156950466)
    )

    //self similarity
    val drmCooc = CooccurrenceAnalysis.cooccurrences(drmARaw = drmA, drmBs = Array(drmB))
    //var cp = drmSelfCooc(0).checkpoint()
    //cp.writeDRM("/tmp/cooc-spark/")//to get values written
    val matrixSelfCooc = drmCooc(0).checkpoint().collect
    val diffMatrix = matrixSelfCooc.minus(matrixLLRCoocAtAControl)
    var n = (new MatrixOps(m = diffMatrix)).norm
    n should be < 1E-10

    //cross similarity
    val matrixCrossCooc = drmCooc(1).checkpoint().collect
    val diff2Matrix = matrixCrossCooc.minus(matrixLLRCoocBtAControl)
    n = (new MatrixOps(m = diff2Matrix)).norm
    n should be < 1E-10
  }

  test("LLR calc") {
    val numInteractionsWithAandB = 10L
    val numInteractionsWithA = 100L
    val numInteractionsWithB = 200L
    val numInteractions = 10000l

    val llr = CooccurrenceAnalysis.loglikelihoodRatio(numInteractionsWithA, numInteractionsWithB, numInteractionsWithAandB, numInteractions)

    assert(llr == 17.19462327013025)
  }

  test("downsampling by number per row") {
    val a = dense((1, 1, 1, 1, 0),
      (1, 1, 1, 1, 1),
      (0, 0, 0, 0, 1),
      (1, 1, 0, 1, 0)
    )
    val drmA: DrmLike[Int] = drmParallelize(m = a, numPartitions = 2)

    val downSampledDrm = CooccurrenceAnalysis.sampleDownAndBinarize(drmA, 0xdeadbeef, 4)
    //count non-zero values, should be == 7
    var numValues = 0
    val m = downSampledDrm.collect
    val it = m.iterator()
    while (it.hasNext) {
      val v = it.next().vector()
      val nonZeroIt = v.nonZeroes().iterator()
      while (nonZeroIt.hasNext) {
        numValues += 1
        nonZeroIt.next()
      }
    }

    assert(numValues == 8) //Don't change the random seed or this may fail.
  }
}
