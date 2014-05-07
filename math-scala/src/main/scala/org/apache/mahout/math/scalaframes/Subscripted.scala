package org.apache.mahout.math.scalaframes

/** Indexed something (row, column) */
class Subscripted {

  var name: Option[String] = None
  var ordinal:Option[Int] = None

  /** Implies column removal in select(). */
  var del:Boolean = false

  def this(name:String) {
    this()
    this.name = Some(name)
  }

  def this(ordinal: Int) {
    this()
    if (ordinal < 0) {
      this.ordinal = Some(-ordinal)
      this.del = true
    } else {
      this.ordinal = Some(ordinal)
    }
  }

  def unary_-():Subscripted = {
    del = !del
    this
  }

}
