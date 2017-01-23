package org.apache.mahout.math.backend

import org.apache.mahout.math.backend.jvm.JvmBackend

import collection._
import scala.reflect.{ClassTag, classTag}
import jvm.JvmBackend

/**
  * == Overview ==
  *
  * Backend representing collection of in-memory solvers or distributed operators.
  *
  * == Note to implementors ==
  *
  * Backend is expected to initialize & verify its own viability lazily either upon first time the
  * class is loaded, or upon the first invocation of any of its methods. After that, the value of
  * [[Backend.isAvailable]] must be cached and defined.
  *
  * A Backend is also a [[SolverFactory]] of course in a sense that it enumerates solvers made
  * available via the backend.
  */
trait Backend extends SolverFactory {

  /**
    * If backend has loaded (lazily) ok and verified its availability/functionality,
    * this must return `true`.
    *
    * @return `true`
    */
  def isAvailable: Boolean

}
