package org.apache.mahout.math.scalaframes

class DplyrOps(private[scalaframes] val frame:DFrameLike) {

  // TODO
  def select(selections: Subscripted*): DFrameLike = this

  def mutate(mutations: LHS*): DFrameLike = this

}
