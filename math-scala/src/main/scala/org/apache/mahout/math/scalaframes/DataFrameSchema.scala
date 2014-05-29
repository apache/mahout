package org.apache.mahout.math.scalaframes

class DataFrameSchema(val cols: Iterable[(String, DFType.DFType)]) extends Serializable

object DataFrameSchema {

  def fromTSVLine(line: String): DataFrameSchema = new DataFrameSchema(
    line.split("\t").view.filter(_.length > 0).map(_ -> DFType.string))

  def toTSVLine(dfs: DataFrameSchema) = {
    val names = dfs.cols.map(_._1)
    if (names.size == 0) "" else names.tail.scanLeft(names.head)(_ + '\t' + _)
  }
}