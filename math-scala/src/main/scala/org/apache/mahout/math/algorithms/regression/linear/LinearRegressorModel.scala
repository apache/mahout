package org.apache.mahout.math.algorithms.regression.linear

import org.apache.mahout.math.algorithms.regression.RegressorModel
import org.apache.mahout.math.Vector

trait LinearRegressorModel[K] extends RegressorModel[K] {

  var beta: Vector = _
  var se: Vector = _
  var tScore: Vector = _
  var pval: Vector = _



}
