package org.apache.mahout.math

package object scalaframes {

  implicit def frame2Dplyr(f:DataFrameLike):DplyrLikeOps = new DplyrLikeOps(f)
  implicit def dplyr2Frame(dplyr:DplyrLikeOps):DataFrameLike = dplyr.frame

  implicit def lhs(name:String):LHS = new LHS(name)
  implicit def nf(name:String):Subscripted = new Subscripted(name)
  implicit def nf(ordinal:Int):Subscripted = new Subscripted(ordinal)

  implicit def i2ops(x:Int) = new IntCellOps(x)
  implicit def d2ops(x:Double) = new DoubleCellOps(x)
  implicit def l2ops(x:Long) = new LongCellOps(x)

  def col(nf:Subscripted):CellOps = null

  /** Data Frame object type */
  object DFType extends Enumeration {
    val int64, long, double, string, bytes = Value
    type DFType = Value
  }

}
