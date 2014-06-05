package org.apache.mahout.math.scalaframes

class IntCellOps(val x:Int) extends AnyVal with CellOps {

  def +(that: CellOps): CellOps = x + that.toInt

  def -(that: CellOps): CellOps = x - that.toInt

  def unary_-(): CellOps = -x

  def *(that: CellOps): CellOps = x * that.toInt

  def /(that: CellOps): CellOps = x / that.toInt

  def toLong: Long = x.toLong

  def toInt: Int = x

  def toDouble: Double = x.toDouble
}
