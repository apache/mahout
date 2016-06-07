package org.apache.mahout.javaCppTest

import org.scalatest.{FunSuite, Matchers}


class HelloNativeTestSuite extends FunSuite with Matchers {

  test("HelloNative"){
    val nTest = new HelloNative

    nTest.set_property("Hello Native")
    assert(nTest.get_property() == "Hello Native")
  }

}
