package org.apache.mahout.test

import org.apache.mahout.math.drm.DistributedContext
import org.scalatest.{Suite, FunSuite, Matchers}

/**
 * Unit tests that use a distributed context to run
 */
trait DistributedMahoutSuite extends MahoutSuite  { this: Suite =>
  protected implicit var mahoutCtx: DistributedContext
}
