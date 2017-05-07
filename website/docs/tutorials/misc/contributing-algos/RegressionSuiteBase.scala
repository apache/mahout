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

package org.apache.mahout.math.algorithms

import org.apache.mahout.math.algorithms.regression._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}

trait RegressionSuiteBase extends DistributedMahoutSuite with Matchers {
  this: FunSuite =>

  val epsilon = 1E-6

  test("ordinary least squares") {
    /*
    R Prototype:
    dataM <- matrix( c(2, 2, 10.5, 10, 29.509541,
      1, 2, 12,   12, 18.042851,
      1, 1, 12,   13, 22.736446,
      2, 1, 11,   13, 32.207582,
      1, 2, 12,   11, 21.871292,
      2, 1, 16,   8,  36.187559,
      6, 2, 17,   1,  50.764999,
      3, 2, 13,   7,  40.400208,
      3, 3, 13,   4,  45.811716), nrow=9, ncol=5, byrow=TRUE)


    X = dataM[, c(1,2,3,4)]
    y = dataM[, c(5)]

    model <- lm(y ~ X )
    summary(model)

     */

    val drmData = drmParallelize(dense(
      (2, 2, 10.5, 10, 29.509541),  // Apple Cinnamon Cheerios
      (1, 2, 12,   12, 18.042851),  // Cap'n'Crunch
      (1, 1, 12,   13, 22.736446),  // Cocoa Puffs
      (2, 1, 11,   13, 32.207582),  // Froot Loops
      (1, 2, 12,   11, 21.871292),  // Honey Graham Ohs
      (2, 1, 16,   8,  36.187559),  // Wheaties Honey Gold
      (6, 2, 17,   1,  50.764999),  // Cheerios
      (3, 2, 13,   7,  40.400208),  // Clusters
      (3, 3, 13,   4,  45.811716)), numPartitions = 2)


    val drmX = drmData(::, 0 until 4)
    val drmY = drmData(::, 4 until 5)

    val model = new OrdinaryLeastSquares[Int]().fit(drmX, drmY, 'calcCommonStatistics → false)

    val estimate = model.beta
    val Ranswers = dvec(-1.336265, -13.157702, -4.152654, -5.679908, 163.179329)

    val epsilon = 1E-6
    (estimate - Ranswers).sum should be < epsilon

    // TODO add test for S.E / pvalue
  }

  test("cochrane-orcutt"){
    /* R Prototype:
    library(orcutt)

    df = data.frame(t(data.frame(
        c(20.96,  127.3),
        c(21.40,  130.0),
        c(21.96,  132.7),
        c(21.52,  129.4),
        c(22.39,  135.0),
        c(22.76,  137.1),
        c(23.48,  141.2),
        c(23.66,  142.8),
        c(24.10,  145.5),
        c(24.01,  145.3),
        c(24.54,  148.3),
        c(24.30,  146.4),
        c(25.00,  150.2),
        c(25.64,  153.1),
        c(26.36,  157.3),
        c(26.98,  160.7),
        c(27.52,  164.2),
        c(27.78,  165.6),
        c(28.24,  168.7),
        c(28.78,  171.7))))

    rownames(df) <- NULL
    colnames(df) <- c("y", "x")
    my_lm = lm(y ~ x, data=df)
    coch = cochrane.orcutt(my_lm)

    ///////////////////////////////////////
    The R-implementation is kind of...silly.

    The above works- converges at 318 iterations- the transformed DW is   1.72, yet the rho is
     .95882.   After 318 iteartions, this will also report a rho of .95882 (which sugguests SEVERE
     autocorrelation- nothing close to 1.72.

     At anyrate, the real prototype for this is the example from Applied Linear Statistcal Models
     5th Edition by Kunter, Nachstheim, Neter, and Li.  They also provide some interesting notes on p 494:
     1) "Cochrane-Orcutt does not always work properly.  A major reason is that when the error terms
     are positively autocorrelated, the estimate r in (12.22) tends to underestimate the autocorrelation
     parameter rho.  When this bias is serious, it can significantly reduce the effectiveness of the
     Cochrane-Orcutt approach.
     2. There exists an approximate relation between the Durbin Watson test statistic D in (12.14)
     and the estimated autocorrelation paramater r in (12.22):
     D ~= 2(1-r)"

     They also note on p492:
     "... If the process does not terminate after one or two iterations, a different procedure
     should be employed."
     This differs from the logic found elsewhere, and the method presented in R where, in the simple
      example in the prototype, the procedure runs for 318 iterations. This is why the default
     maximum iteratoins are 3, and should be left as such.

     Also, the prototype and 'correct answers' are based on the example presented in Kunter et. al on
     p492-4 (including dataset).

     */

    val alsmBlaisdellCo = drmParallelize( dense(
      (20.96,  127.3),
      (21.40,  130.0),
      (21.96,  132.7),
      (21.52,  129.4),
      (22.39,  135.0),
      (22.76,  137.1),
      (23.48,  141.2),
      (23.66,  142.8),
      (24.10,  145.5),
      (24.01,  145.3),
      (24.54,  148.3),
      (24.30,  146.4),
      (25.00,  150.2),
      (25.64,  153.1),
      (26.36,  157.3),
      (26.98,  160.7),
      (27.52,  164.2),
      (27.78,  165.6),
      (28.24,  168.7),
      (28.78,  171.7) ))

    val drmY = alsmBlaisdellCo(::, 0 until 1)
    val drmX = alsmBlaisdellCo(::, 1 until 2)

    var coModel = new CochraneOrcutt[Int]().fit(drmX, drmY , ('iterations -> 2))
    val coResiduals = drmY - coModel.predict(drmX)

    val correctRho = 0.631166
    (coModel.rhos(1) - correctRho) should be < epsilon

    val shortEpsilon = 1E-4 // book rounded off pretty short
    val correctBeta = dvec(0.17376, -1.0685)
    (coModel.betas(1) - correctBeta).sum.abs < shortEpsilon

    val correctSe = dvec(0.002957, 0.45332)
    (coModel.se - correctSe).sum.abs < shortEpsilon
  }

  test("foo") {
    import org.apache.mahout.math.algorithms.regression.Foo

    val drmA = drmParallelize(dense((1.0, 1.2, 1.3, 1.4),
                                    (1.1, 1.5, 2.5, 1.0),
                                    (6.0, 5.2, -5.2, 5.3),
                                    (7.0,6.0, 5.0, 5.0),
                                    (10.0, 1.0, 20.0, -10.0)))

    val model = new Foo().fit(drmA(::, 0 until 2), drmA(::, 2 until 3), 'guessThisNumber -> 2.0)

    val myAnswer = model.predict(drmA).collect
    val correctAnswer = dense( (2.0),
                                (2.0),
                                (2.0),
                                (2.0),
                                (2.0))


    val epsilon = 1E-6
    (myAnswer - correctAnswer).sum should be < epsilon
  }
}
