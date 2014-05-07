package org.apache.mahout.math.scalaframes

class DplyrOps(private[scalaframes] val frame:DFrameLike) {

  // TODO
  def select(selections: NamedFragment*): DplyrOps = this

  def mutate(mutations: LHS*): DplyrOps = this

}
