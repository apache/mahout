package org.apache.mahout.math.scalaframes

class DoubleCellOps(val x:Double) extends AnyVal with CellOps {

  def +(that: CellOps): CellOps = x + that.toDouble

  def -(that: CellOps): CellOps = x - that.toDouble

  def unary_-(): CellOps = -x

  def *(that: CellOps): CellOps = x * that.toDouble

  def /(that: CellOps): CellOps = x / that.toDouble

  def toLong: Long = x.toLong

  def toInt: Int = x.toInt

  def toDouble: Double = x
}