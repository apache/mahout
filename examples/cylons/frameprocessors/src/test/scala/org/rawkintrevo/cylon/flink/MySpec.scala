package org.apache.mahout.cylon-example.flink

import org.specs._
import org.specs.runner.{ConsoleRunner, JUnit4}

class MySpecTest extends JUnit4(MySpec)
//class MySpecSuite extends ScalaTestSuite(MySpec)
object MySpecRunner extends ConsoleRunner(MySpec)

object MySpec extends Specification {
  "This wonderful system" should {
    "save the world" in {
      val list = Nil
      list must beEmpty
    }
  }
}
