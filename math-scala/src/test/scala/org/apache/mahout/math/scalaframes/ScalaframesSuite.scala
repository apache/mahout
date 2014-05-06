package org.apache.mahout.math.scalaframes

import org.scalatest.FunSuite
import org.apache.mahout.test.MahoutSuite

class ScalaframesSuite extends FunSuite with MahoutSuite {

  test("select") {
    import org.apache.mahout.math.scalaframes._
    val testFrame = new BaseDFrame()

//    testFrame.select()
    val selectedFrame = testFrame.mutate(
      "ACol" := col("5") + col("4"),
      "BCol" := col("AAA") + 3
    )



  }

}
