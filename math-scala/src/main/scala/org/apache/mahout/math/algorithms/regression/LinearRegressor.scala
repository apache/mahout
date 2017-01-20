package org.apache.mahout.math.algorithms.regression

import org.apache.mahout.math.{Vector => MahoutVector}

trait LinearRegressor extends Regressor {

  var beta: MahoutVector = _
  var se: MahoutVector = _
  var tScore: MahoutVector = _
  var pval: MahoutVector = _
  var degreesFreedom: Int = _

}
