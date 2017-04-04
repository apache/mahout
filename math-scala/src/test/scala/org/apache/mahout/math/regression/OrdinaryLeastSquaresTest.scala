package org.apache.mahout.math.regression

import org.apache.mahout.math.algorithms.regression.{OrdinaryLeastSquares, OrdinaryLeastSquaresModel}
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm.{CheckpointedDrm, drmParallelize}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.test.DistributedMahoutSuite
import org.apache.mahout.test.MahoutSuite
import org.scalatest.{FunSuite, Matchers}
import collection._
import JavaConversions._
import collection.JavaConversions
/**
  * Created by dustinvanstee on 3/30/17.
  */
trait OrdinaryLeastSquaresTest extends DistributedMahoutSuite with Matchers {
  this: FunSuite =>

  val epsilon = 1E-3
  test("Simple Small Linear Model1") {

    // Sample Cereal Data ...
    val inCoreData = dense(
      (2, 2, 10.5, 10, 29.509541), // Apple Cinnamon Cheerios
      (1, 2, 12, 12, 18.042851), // Cap'n'Crunch
      (1, 1, 12, 13, 22.736446), // Cocoa Puffs
      (2, 1, 11, 13, 32.207582), // Froot Loops
      (1, 2, 12, 11, 21.871292), // Honey Graham Ohs
      (2, 1, 16, 8, 36.187559), // Wheaties Honey Gold
      (6, 2, 17, 1, 50.764999), // Cheerios
      (3, 2, 13, 7, 40.400208), // Clusters
      (3, 3, 13, 4, 45.811716))


    val drmData: CheckpointedDrm[Int] = drmParallelize(inCoreData, numPartitions = 2)
    val drmX = drmData(::, 0 until 4)
    val y = drmData(::, 4 until 5)
    val model: OrdinaryLeastSquaresModel[Int] = new OrdinaryLeastSquares[Int]().fit(drmX, y)
    //println(model.beta.toString)
    assert(model.se(0) === 2.6878127323908942)
    model.beta(4) - 163.179 should be < epsilon
    model.beta(0) - (-1.336) should be < epsilon
    model.beta(1) - (-13.15770) should be < epsilon
    model.beta(2) - (-4.15265) should be < epsilon
    model.beta(3) - (-5.679908) should be < epsilon

    model.tScore(0) - (-0.49715717) should be < epsilon
    model.tScore(1) - (-2.43932888) should be < epsilon
    model.tScore(2) - (-2.32654000) should be < epsilon
    model.tScore(3) - (-3.01022444) should be < epsilon
    model.tScore(4) -  3.143183937  should be < epsilon

    model.fScore - 16.38542361  should be < epsilon
    //println(model.summary)

  }

}