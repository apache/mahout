/**
  * Licensed to the Apache Software Foundation (ASF) under one
  * or more contributor license agreements. See the NOTICE file
  * distributed with this work for additional information
  * regarding copyright ownership. The ASF licenses this file
  * to you under the Apache License, Version 2.0 (the
  * "License"); you may not use this file except in compliance
  * with the License. You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing,
  * software distributed under the License is distributed on an
  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  * KIND, either express or implied. See the License for the
  * specific language governing permissions and limitations
  * under the License.
  */

package org.apache.mahout.math.algorithms.regression

import org.apache.mahout.math.algorithms.regression.tests.FittnessTests
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings.dvec
import org.apache.mahout.math.{Matrix, Vector => MahoutVector}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.commons.math3.distribution._

import scala.language.higherKinds

trait LinearRegressorModel[K] extends RegressorModel[K] {

  var beta: MahoutVector = _
  var se: MahoutVector = _
  var tScore: MahoutVector = _
  var pval: MahoutVector = _
  var degreesFreedom: Int = _
  var trainingExamples :Int = _
  var fScore: Double = _
  var fpval: Double = _


}

trait LinearRegressorFitter[K] extends RegressorFitter[K] {

  var calcStandardErrors: Boolean = _
  var calcCommonStatistics: Boolean = _

  def fit(drmX: DrmLike[K],
          drmTarget: DrmLike[K],
          hyperparameters: (Symbol, Any)*): LinearRegressorModel[K]


  def setStandardHyperparameters(hyperparameters: Map[Symbol, Any] = Map('foo -> None)): Unit = {
    calcCommonStatistics = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('calcCommonStatistics, true)
    calcStandardErrors = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('calcStandardErrors, true)
    addIntercept = hyperparameters.asInstanceOf[Map[Symbol, Boolean]].getOrElse('addIntercept, true)
  }

  def calculateStandardError[M[K] <: LinearRegressorModel[K]](X: DrmLike[K],
                             drmTarget: DrmLike[K],
                             drmXtXinv: Matrix,
                             model: M[K]): M[K] = {
    import org.apache.mahout.math.function.Functions.SQRT
    import org.apache.mahout.math.scalabindings.MahoutCollections._
    var modelOut = model
    val yhat = X %*% model.beta
    val residuals = drmTarget - yhat
    val ete = (residuals.t %*% residuals).collect // 1x1
    val n = drmTarget.nrow
    val k = safeToNonNegInt(X.ncol)
    val invDegFreedomKindOf = 1.0 / (n - k)
    val varCovarMatrix = invDegFreedomKindOf * ete(0,0) * drmXtXinv
    val se = varCovarMatrix.viewDiagonal.assign(SQRT)
    val tScore = model.beta / se
    val tDist = new TDistribution(n-k)

    // This is the residual sum of squares for just the intercept
    //println(" drmTarget.ncol) = " +  drmTarget.ncol)
    val interceptCol = model.beta.length - 1
    val targetMean: Double = drmTarget.colMeans().get(0)
    val rssint: Double = ((drmTarget - targetMean  ).t %*% (drmTarget - targetMean)).zSum()
    val rssmod: Double = ete.zSum()
    // ete above is the RSS for the calculated model

    //println(" model.beta(0) = " +  model.beta(0))
    //println(" model.beta(interceptCol) = " +  model.beta(interceptCol))
    //println("rssint = " + rssint)
    //println("rssmod = " + rssmod)

    val groupDof = X.ncol-1
    val fScore = ((rssint - rssmod) / groupDof) / ( rssmod / (X.nrow - groupDof- 1 ))
    //println("groupDof = " + groupDof)
    //println("fScore = " + fScore)
    val fDist = new FDistribution(groupDof,n-groupDof-1)


    val pval = dvec(tScore.toArray.map(t => 2 * (1.0 - tDist.cumulativeProbability(Math.abs(t))) ))
    val fpval = 1.0 - fDist.cumulativeProbability(fScore)

    // ^^ TODO bug in this calculation- fix and add test
    //degreesFreedom = k
    modelOut.se = se
    modelOut.tScore = tScore
    modelOut.pval = pval
    // for degrees of freedom, dont count the intercept term that was added
    modelOut.degreesFreedom = X.ncol - 1
    modelOut.fScore = fScore
    modelOut.trainingExamples = n.toInt
    modelOut.fpval = fpval
    modelOut.summary = generateSummaryString(modelOut)

    if (calcCommonStatistics){
      modelOut = calculateCommonStatistics(modelOut, drmTarget, residuals)
    }
    modelOut
  }

  def calculateCommonStatistics[M[K] <: LinearRegressorModel[K]](model: M[K],
                                                                 drmTarget: DrmLike[K],
                                                                 residuals: DrmLike[K]): M[K] ={
    var modelOut = model
    modelOut = FittnessTests.CoefficientOfDetermination(model, drmTarget, residuals)
    modelOut = FittnessTests.MeanSquareError(model, residuals)

    modelOut.summary += s"MSE: ${modelOut.mse}\nR2: ${modelOut.r2}\n"
    modelOut
  }

  def modelPostprocessing[M[K] <: LinearRegressorModel[K]](model: M[K],
                     X: DrmLike[K],
                     drmTarget: DrmLike[K],
                     drmXtXinv: Matrix): M[K] = {
    var modelOut = model
    if (calcStandardErrors) {
      modelOut = calculateStandardError(X, drmTarget, drmXtXinv, model )
    } else {
      modelOut.summary = "Coef.\t\tEstimate\n" +
        (0 until X.ncol).map(i => s"X${i}\t${modelOut.beta(i)}").mkString("\n")
      if (calcCommonStatistics) { // we do this in calcStandard errors to avoid calculating residuals twice
        val residuals = drmTarget - (X %*% modelOut.beta)
        modelOut = calculateCommonStatistics(modelOut, drmTarget, residuals)
      }

      modelOut
    }

    if (addIntercept) {
      model.summary.replace(s"X${X.ncol - 1}", "(Intercept)")
      model.addIntercept = true
    }
    model
  }

  def generateSummaryString[M[K] <: LinearRegressorModel[K]](model: M[K]): String = {

    /*  Model after R implementation ...
    Call:
      lm(formula = target ~ a + b + c + d, data = df1)

    Residuals:
    1       2       3       4       5       6       7       8       9
    -4.2799  0.5059 -2.2783  4.3765 -1.3455  0.7202 -1.8063  1.2889  2.8184

    Coefficients:
      Estimate Std. Error t value Pr(>|t|)
    (Intercept)  163.179     51.915   3.143   0.0347 *
      a             -1.336      2.688  -0.497   0.6452
      b            -13.158      5.394  -2.439   0.0713 .
      c             -4.153      1.785  -2.327   0.0806 .
      d             -5.680      1.887  -3.010   0.0395 *
      ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    */

    val k = model.beta.length

      // Using Formatted Print here to pretty print the columns
      "\nCoef.\t\tEstimate\t\tStd. Error\t\tt-score\t\t\tPr(Beta=0)\n" +
      (0 until k).map(i => "X%-3d\t\t%+5.5f\t\t%+5.5f\t\t%+5.5f\t\t%+5.5f".format(i,model.beta(i),model.se(i),model.tScore(i),model.pval(i))).mkString("\n") +
      "\nF-statistic: " + model.fScore + " on " + model.degreesFreedom + " and " +
        (model.trainingExamples - model.degreesFreedom  - 1) + " DF,  p-value: " + 0.009545 + "\n"

  }
}
