package org.apache.mahout.math.scalaframes

/**
 * "left-hand side" fragment
 */
class LHS(val name: String) {

  private[scalaframes] var assignmentFunc: () => Any = _

  def :=(af: => Any): this.type = {
    assignmentFunc = () => af
    this
  }


}
