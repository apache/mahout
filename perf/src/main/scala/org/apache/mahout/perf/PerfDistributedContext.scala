package org.apache.mahout.perf

/**
  * Created by skanjila on 6/20/16.
  */
import org.apache.mahout.math.drm.{DistributedContext, DistributedEngine, _}

trait PerfDistributedContext extends DistributedContext {

  def close(): Unit = return

  def engine = DistributedEngine
}