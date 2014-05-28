package org.apache.mahout.math.scalaframes

trait CellOps extends Any {

  def +(that:CellOps):CellOps
  def -(that:CellOps):CellOps
  def unary_-():CellOps
  def *(that:CellOps):CellOps
  def /(that:CellOps):CellOps

  def toLong:Long
  def toInt:Int
  def toDouble:Double
  def toString:String


}
