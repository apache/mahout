package org.apache.mahout.math.algorithms.regression

import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.{Vector => MahoutVector}

trait LinearRegressorModel[K] extends RegressorModel[K] {

  var beta: MahoutVector = _
  var se: MahoutVector = _
  var tScore: MahoutVector = _
  var pval: MahoutVector = _
  var degreesFreedom: Int = _

}

trait LinearRegressorModelFactory[K] extends RegressorModelFactory[K] {

  def fit(drmX: DrmLike[K],
          drmTarget: DrmLike[K],
          hyperparameters: (Symbol, Any)*): LinearRegressorModel[K]

}
