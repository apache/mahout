package org.apache.mahout.math.scalaframes

/**
 *
 * @author dmitriy
 */
class NamedFragment(val name:String) {

  // Implies column removal in select().
  var del:Boolean = false

  def unary_-():NamedFragment = {
    del = !del
    this
  }

}
