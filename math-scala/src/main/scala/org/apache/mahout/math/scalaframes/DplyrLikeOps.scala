package org.apache.mahout.math.scalaframes

class DplyrLikeOps(private[scalaframes] val frame:DataFrameLike) {

  // TODO
  def select(selections: Subscripted*): DataFrameLike = this

  def mutate(mutations: LHS*): DataFrameLike = this

}
