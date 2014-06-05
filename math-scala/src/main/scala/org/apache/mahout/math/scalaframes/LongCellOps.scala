package org.apache.mahout.math.scalaframes

class LongCellOps(val x:Long) extends AnyVal with CellOps {

  def +(that: CellOps): CellOps = x + that.toLong

  def -(that: CellOps): CellOps = x - that.toLong

  def unary_-(): CellOps = -x

  def *(that: CellOps): CellOps = x * that.toLong

  def /(that: CellOps): CellOps = x / that.toLong

  def toLong: Long = x

  def toInt: Int = x.toInt

  def toDouble: Double = x.toDouble
}

