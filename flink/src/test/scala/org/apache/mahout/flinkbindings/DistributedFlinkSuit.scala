package org.apache.mahout.flinkbindings

import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.Suite
import org.apache.mahout.math.drm.DistributedContext
import org.apache.flink.api.java.ExecutionEnvironment

trait DistributedFlinkSuit extends DistributedMahoutSuite { this: Suite =>

  protected implicit var mahoutCtx: DistributedContext = _
  protected var env: ExecutionEnvironment = null
  
  def initContext() {
    env = ExecutionEnvironment.getExecutionEnvironment
    mahoutCtx = env
  }

  override def beforeEach() {
    initContext()
  }

  override def afterEach() {
    super.afterEach()
//    env.execute("Mahout Flink Binding Test Suite")
  }

}