package org.apache.mahout.math

/**
 *
 * @author dmitriy
 */
package object scalaframes {

  implicit def frame2Dplyr(f:DFrameLike):DplyrOps = new DplyrOps(f)
  implicit def dplyr2Frame(dplyr:DplyrOps):DFrameLike = dplyr.frame

  implicit def lhs(name:String):LHS = new LHS(name)
  implicit def nf(name:String):NamedFragment = new NamedFragment(name)

  def col(nf:NamedFragment):CellOps = null


}
