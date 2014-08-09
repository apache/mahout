/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.math

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.DrmLike

/**
 * This package holds all decomposition and factorization-like methods, all that we were able to make
 * distributed engine-independent so far, anyway.
 */
package object decompositions {

  // ================ In-core decompositions ===================

  /**
   * In-core SSVD algorithm.
   *
   * @param a input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations
   * @return (U,V,s)
   */
  def ssvd(a: Matrix, k: Int, p: Int = 15, q: Int = 0) = SSVD.ssvd(a, k, p, q)

  /**
   * PCA based on SSVD that runs without forming an always-dense A-(colMeans(A)) input for SVD. This
   * follows the solution outlined in MAHOUT-817. For in-core version it, for most part, is supposed
   * to save some memory for sparse inputs by removing direct mean subtraction.<P>
   *
   * Hint: Usually one wants to use AV which is approsimately USigma, i.e.<code>u %*%: diagv(s)</code>.
   * If retaining distances and orignal scaled variances not that important, the normalized PCA space
   * is just U.
   *
   * Important: data points are considered to be rows.
   *
   * @param a input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations
   * @return (U,V,s)
   */
  def spca(a: Matrix, k: Int, p: Int = 15, q: Int = 0) =
    SSVD.spca(a = a, k = k, p = p, q = q)

  // ============== Distributed decompositions ===================

  /**
   * Distributed _thin_ QR. A'A must fit in a memory, i.e. if A is m x n, then n should be pretty
   * controlled (<5000 or so). <P>
   *
   * It is recommended to checkpoint A since it does two passes over it. <P>
   *
   * It also guarantees that Q is partitioned exactly the same way (and in same key-order) as A, so
   * their RDD should be able to zip successfully.
   */
  def dqrThin[K: ClassTag](drmA: DrmLike[K], checkRankDeficiency: Boolean = true): (DrmLike[K], Matrix) =
    DQR.dqrThin(drmA, checkRankDeficiency)

  /**
   * Distributed Stochastic Singular Value decomposition algorithm.
   *
   * @param drmA input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations
   * @return (U,V,s). Note that U, V are non-checkpointed matrices (i.e. one needs to actually use them
   *         e.g. save them to hdfs in order to trigger their computation.
   */
  def dssvd[K: ClassTag](drmA: DrmLike[K], k: Int, p: Int = 15, q: Int = 0):
  (DrmLike[K], DrmLike[Int], Vector) = DSSVD.dssvd(drmA, k, p, q)

  /**
   * Distributed Stochastic PCA decomposition algorithm. A logical reflow of the "SSVD-PCA options.pdf"
   * document of the MAHOUT-817.
   *
   * @param drmA input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations (hint: use either 0 or 1)
   * @return (U,V,s). Note that U, V are non-checkpointed matrices (i.e. one needs to actually use them
   *         e.g. save them to hdfs in order to trigger their computation.
   */
  def dspca[K: ClassTag](drmA: DrmLike[K], k: Int, p: Int = 15, q: Int = 0):
  (DrmLike[K], DrmLike[Int], Vector) = DSPCA.dspca(drmA, k, p, q)

  /** Result for distributed ALS-type two-component factorization algorithms */
  type FactorizationResult[K] = ALS.Result[K]

  /** Result for distributed ALS-type two-component factorization algorithms, in-core matrices */
  type FactorizationResultInCore = ALS.InCoreResult
  
  /**
   * Run ALS.
   * <P>
   *
   * Example:
   *
   * <pre>
   * val (u,v,errors) = als(input, k).toTuple
   * </pre>
   *
   * ALS runs until (rmse[i-1]-rmse[i])/rmse[i-1] < convergenceThreshold, or i==maxIterations,
   * whichever earlier.
   * <P>
   *
   * @param drmA The input matrix
   * @param k required rank of decomposition (number of cols in U and V results)
   * @param convergenceThreshold stop sooner if (rmse[i-1] - rmse[i])/rmse[i - 1] is less than this
   *                             value. If <=0 then we won't compute RMSE and use convergence test.
   * @param lambda regularization rate
   * @param maxIterations maximum iterations to run regardless of convergence
   * @tparam K row key type of the input (100 is probably more than enough)
   * @return { @link org.apache.mahout.math.drm.decompositions.ALS.Result}
   */
  def dals[K: ClassTag](
      drmA: DrmLike[K],
      k: Int = 50,
      lambda: Double = 0.0,
      maxIterations: Int = 10,
      convergenceThreshold: Double = 0.10
      ): FactorizationResult[K] =
    ALS.dals(drmA, k, lambda, maxIterations, convergenceThreshold)

}
