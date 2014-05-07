package org.apache.mahout.math

/**
 *
 * @author dmitriy
 */
package object scalaframes {

  implicit def frame2Dplyr(f:DFrameLike):DplyrOps = new DplyrOps(f)
  implicit def dplyr2Frame(dplyr:DplyrOps):DFrameLike = dplyr.frame

  implicit def lhs(name:String):LHS = new LHS(name)
  implicit def nf(name:String):Subscripted = new Subscripted(name)
  implicit def nf(ordinal:Int):Subscripted = new Subscripted(ordinal)

  def col(nf:Subscripted):CellOps = null


}
