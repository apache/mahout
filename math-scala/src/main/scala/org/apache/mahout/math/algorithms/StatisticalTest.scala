package org.apache.mahout.math.algorithms

/**
  * Created by rawkintrevo on 1/20/17.
  */
trait StatisticalTest extends Model {

  def test(model: Model): Model

}
