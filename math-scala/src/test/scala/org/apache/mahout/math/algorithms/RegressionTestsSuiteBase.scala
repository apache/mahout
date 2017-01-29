package org.apache.mahout.math.algorithms

import org.apache.mahout.math.algorithms.regression.{OrdinaryLeastSquares, OrdinaryLeastSquaresModel}
import org.apache.mahout.math.algorithms.regression.tests._
import org.apache.mahout.math.drm.drmParallelize
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.{`::`, dense}
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}


trait RegressionTestsSuiteBase extends DistributedMahoutSuite with Matchers {
  this: FunSuite =>

  val epsilon = 1E-4

  test("fittness tests") {
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

    model <- lm(y ~ X)
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

    //drmData.collect(::, 0 until 4)

    val drmX = drmData(::, 0 until 4)
    val drmY = drmData(::, 4 until 5)

    val model = new OrdinaryLeastSquares[Int]().fit(drmX, drmY)
    model.r2
    model.mse

    // Answers from running similar algorithm in R
    val rR2 = 0.9425
    val rMSE = 6.457157

    val r2 = model.testResults.get('r2).get.asInstanceOf[Double]
    val mse = model.testResults.get('mse).get.asInstanceOf[Double]
    (rR2 - r2) should be < epsilon
    (rMSE - mse) should be < epsilon

//    model.getTestResult[Double]('r2) match {
//      case Some(r2) =>
//      case None => throw new Exception("r2 not found in testResults")
//    }
//
//    model.getTestResult[Double]('mse) match {
//      case Some(mse) => (rMSE - mse) should be < epsilon
//      case None => throw new Exception("mse not found in testResults")
//    }
//
//    // Test that non-existent tests do as expected
//    model.getTestResult[Double]('foo) match {
//      case Some(foo) => throw new Exception("foo existed in testResults ??")
//      case None => 1 should equal(1)
//    }

  }

//  test("autocorrelation tests") {
//    /*
//    R Prototype:
//    install.packages("lmtest", repos='http://cran.us.r-project.org')
//    library("lmtest")
//
//    dataM <- matrix( c(2, 2, 10.5, 10, 29.509541,
//      1, 2, 12,   12, 18.042851,
//      1, 1, 12,   13, 22.736446,
//      2, 1, 11,   13, 32.207582,
//      1, 2, 12,   11, 21.871292,
//      2, 1, 16,   8,  36.187559,
//      6, 2, 17,   1,  50.764999,
//      3, 2, 13,   7,  40.400208,
//      3, 3, 13,   4,  45.811716), nrow=9, ncol=5, byrow=TRUE)
//
//
//    X = dataM[, c(1,2,3,4)]
//    y = dataM[, c(5)]
//
//    model <- lm(y ~ X)
//
//    dwtest(y ~ X)
//     */
//
//    val drmData = drmParallelize(dense(
//      (2, 2, 10.5, 10, 29.509541), // Apple Cinnamon Cheerios
//      (1, 2, 12, 12, 18.042851), // Cap'n'Crunch
//      (1, 1, 12, 13, 22.736446), // Cocoa Puffs
//      (2, 1, 11, 13, 32.207582), // Froot Loops
//      (1, 2, 12, 11, 21.871292), // Honey Graham Ohs
//      (2, 1, 16, 8, 36.187559), // Wheaties Honey Gold
//      (6, 2, 17, 1, 50.764999), // Cheerios
//      (3, 2, 13, 7, 40.400208), // Clusters
//      (3, 3, 13, 4, 45.811716)), numPartitions = 2)
//
//    //drmData.collect(::, 0 until 4)
//
//    val drmX = drmData(::, 0 until 4)
//    val drmY = drmData(::, 4 until 5)
//
//    var model = new OrdinaryLeastSquares[Int]()
//    model.fit(drmX, drmY)
//
//    val rDW = 2.2413
//    ///val dwStat = AutocorrelationTests.DurbinWatson(model).testResults("durbin-watson test statistic").asInstanceOf[Double]
//    //(rDW - dwStat) should be < epsilon
//    1 should be 1
//  }
}
