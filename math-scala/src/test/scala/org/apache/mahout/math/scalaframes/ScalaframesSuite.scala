package org.apache.mahout.math.scalaframes

import org.scalatest.FunSuite
import org.apache.mahout.test.MahoutSuite

class ScalaframesSuite extends FunSuite with MahoutSuite {

  test("mutate") {
    val testFrame = new BaseDFrame()

    val mutatedFrame = testFrame.mutate(
      "ACol" := col("5") + col("4"),
      "BCol" := col("AAA") + 3
    )
  }

  test("select") {
    val testFrame = new BaseDFrame()

    val selectedFrame = testFrame.select(
      "ACol",
      "BCol",
      -"CCol"
    )

  }

}
