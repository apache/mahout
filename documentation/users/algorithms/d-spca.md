<!--
 Licensed to the Apache Software Foundation (ASF) under one or more
 contributor license agreements.  See the NOTICE file distributed with
 this work for additional information regarding copyright ownership.
 The ASF licenses this file to You under the Apache License, Version 2.0
 (the "License"); you may not use this file except in compliance with
 the License.  You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->
---
layout: default
title: Distributed Stochastic PCA

    
---

# Distributed Stochastic PCA


## Intro

Mahout has a distributed implementation of Stochastic PCA[1]. This algorithm computes the exact equivalent of Mahout's dssvd(`\(\mathbf{A-1\mu^\top}\)`) by modifying the `dssvd` algorithm so as to avoid forming `\(\mathbf{A-1\mu^\top}\)`, which would densify a sparse input. Thus, it is suitable for work with both dense and sparse inputs.

## Algorithm

Given an *m* `\(\times\)` *n* matrix `\(\mathbf{A}\)`, a target rank *k*, and an oversampling parameter *p*, this procedure computes a *k*-rank PCA by finding the unknowns in `\(\mathbf{A−1\mu^\top \approx U\Sigma V^\top}\)`:

1. Create seed for random *n* `\(\times\)` *(k+p)* matrix `\(\Omega\)`.
2. `\(\mathbf{s_\Omega \leftarrow \Omega^\top \mu}\)`.
3. `\(\mathbf{Y_0 \leftarrow A\Omega − 1 {s_\Omega}^\top, Y \in \mathbb{R}^{m\times(k+p)}}\)`.
4. Column-orthonormalize `\(\mathbf{Y_0} \rightarrow \mathbf{Q}\)` by computing thin decomposition `\(\mathbf{Y_0} = \mathbf{QR}\)`. Also, `\(\mathbf{Q}\in\mathbb{R}^{m\times(k+p)}, \mathbf{R}\in\mathbb{R}^{(k+p)\times(k+p)}\)`.
5. `\(\mathbf{s_Q \leftarrow Q^\top 1}\)`.
6. `\(\mathbf{B_0 \leftarrow Q^\top A: B \in \mathbb{R}^{(k+p)\times n}}\)`.
7. `\(\mathbf{s_B \leftarrow {B_0}^\top \mu}\)`.
8. For *i* in 1..*q* repeat (power iterations):
    - For *j* in 1..*n* apply `\(\mathbf{(B_{i−1})_{∗j} \leftarrow (B_{i−1})_{∗j}−\mu_j s_Q}\)`.
    - `\(\mathbf{Y_i \leftarrow A{B_{i−1}}^\top−1(s_B−\mu^\top \mu s_Q)^\top}\)`.
    - Column-orthonormalize `\(\mathbf{Y_i} \rightarrow \mathbf{Q}\)` by computing thin decomposition `\(\mathbf{Y_i = QR}\)`.
    - `\(\mathbf{s_Q \leftarrow Q^\top 1}\)`.
    - `\(\mathbf{B_i \leftarrow Q^\top A}\)`.
    - `\(\mathbf{s_B \leftarrow {B_i}^\top \mu}\)`.
9. Let `\(\mathbf{C \triangleq s_Q {s_B}^\top}\)`. `\(\mathbf{M \leftarrow B_q {B_q}^\top − C − C^\top + \mu^\top \mu s_Q {s_Q}^\top}\)`.
10. Compute an eigensolution of the small symmetric `\(\mathbf{M = \hat{U} \Lambda \hat{U}^\top: M \in \mathbb{R}^{(k+p)\times(k+p)}}\)`.
11. The singular values `\(\Sigma = \Lambda^{\circ 0.5}\)`, or, in other words, `\(\mathbf{\sigma_i= \sqrt{\lambda_i}}\)`.
12. If needed, compute `\(\mathbf{U = Q\hat{U}}\)`.
13. If needed, compute `\(\mathbf{V = B^\top \hat{U} \Sigma^{−1}}\)`.
14. If needed, items converted to the PCA space can be computed as `\(\mathbf{U\Sigma}\)`.

## Implementation

Mahout `dspca(...)` is implemented in the mahout `math-scala` algebraic optimizer which translates Mahout's R-like linear algebra operators into a physical plan for both Spark and H2O distributed engines.

    def dspca[K](drmA: DrmLike[K], k: Int, p: Int = 15, q: Int = 0): 
    (DrmLike[K], DrmLike[Int], Vector) = {

        // Some mapBlock() calls need it
        implicit val ktag =  drmA.keyClassTag

        val drmAcp = drmA.checkpoint()
        implicit val ctx = drmAcp.context

        val m = drmAcp.nrow
    	val n = drmAcp.ncol
        assert(k <= (m min n), "k cannot be greater than smaller of m, n.")
        val pfxed = safeToNonNegInt((m min n) - k min p)

        // Actual decomposition rank
        val r = k + pfxed

        // Dataset mean
        val mu = drmAcp.colMeans

        val mtm = mu dot mu

        // We represent Omega by its seed.
        val omegaSeed = RandomUtils.getRandom().nextInt()
        val omega = Matrices.symmetricUniformView(n, r, omegaSeed)

        // This done in front in a single-threaded fashion for now. Even though it doesn't require any
        // memory beyond that is required to keep xi around, it still might be parallelized to backs
        // for significantly big n and r. TODO
        val s_o = omega.t %*% mu

        val bcastS_o = drmBroadcast(s_o)
        val bcastMu = drmBroadcast(mu)

        var drmY = drmAcp.mapBlock(ncol = r) {
            case (keys, blockA) ⇒
                val s_o:Vector = bcastS_o
                val blockY = blockA %*% Matrices.symmetricUniformView(n, r, omegaSeed)
                for (row ← 0 until blockY.nrow) blockY(row, ::) -= s_o
                keys → blockY
        }
                // Checkpoint Y
                .checkpoint()

        var drmQ = dqrThin(drmY, checkRankDeficiency = false)._1.checkpoint()

        var s_q = drmQ.colSums()
        var bcastVarS_q = drmBroadcast(s_q)

        // This actually should be optimized as identically partitioned map-side A'B since A and Q should
        // still be identically partitioned.
        var drmBt = (drmAcp.t %*% drmQ).checkpoint()

        var s_b = (drmBt.t %*% mu).collect(::, 0)
        var bcastVarS_b = drmBroadcast(s_b)

        for (i ← 0 until q) {

            // These closures don't seem to live well with outside-scope vars. This doesn't record closure
            // attributes correctly. So we create additional set of vals for broadcast vars to properly
            // create readonly closure attributes in this very scope.
            val bcastS_q = bcastVarS_q
            val bcastMuInner = bcastMu

            // Fix Bt as B' -= xi cross s_q
            drmBt = drmBt.mapBlock() {
                case (keys, block) ⇒
                    val s_q: Vector = bcastS_q
                    val mu: Vector = bcastMuInner
                    keys.zipWithIndex.foreach {
                        case (key, idx) ⇒ block(idx, ::) -= s_q * mu(key)
                    }
                    keys → block
            }

            drmY.uncache()
            drmQ.uncache()

            val bCastSt_b = drmBroadcast(s_b -=: mtm * s_q)

            drmY = (drmAcp %*% drmBt)
                // Fix Y by subtracting st_b from each row of the AB'
                .mapBlock() {
                case (keys, block) ⇒
                    val st_b: Vector = bCastSt_b
                    block := { (_, c, v) ⇒ v - st_b(c) }
                    keys → block
            }
            // Checkpoint Y
            .checkpoint()

            drmQ = dqrThin(drmY, checkRankDeficiency = false)._1.checkpoint()

            s_q = drmQ.colSums()
            bcastVarS_q = drmBroadcast(s_q)

            // This on the other hand should be inner-join-and-map A'B optimization since A and Q_i are not
            // identically partitioned anymore.
            drmBt = (drmAcp.t %*% drmQ).checkpoint()

            s_b = (drmBt.t %*% mu).collect(::, 0)
            bcastVarS_b = drmBroadcast(s_b)
        }

        val c = s_q cross s_b
        val inCoreBBt = (drmBt.t %*% drmBt).checkpoint(CacheHint.NONE).collect -=:
            c -=: c.t +=: mtm *=: (s_q cross s_q)
        val (inCoreUHat, d) = eigen(inCoreBBt)
        val s = d.sqrt

        // Since neither drmU nor drmV are actually computed until actually used, we don't need the flags
        // instructing compute (or not compute) either of the U,V outputs anymore. Neat, isn't it?
        val drmU = drmQ %*% inCoreUHat
        val drmV = drmBt %*% (inCoreUHat %*% diagv(1 / s))

        (drmU(::, 0 until k), drmV(::, 0 until k), s(0 until k))
    }

## Usage

The scala `dspca(...)` method can easily be called in any Spark, Flink, or H2O application built with the `math-scala` library and the corresponding `Spark`, `Flink`, or `H2O` engine module as follows:

    import org.apache.mahout.math._
    import decompositions._
    import drm._
    
    val (drmU, drmV, s) = dspca(drmA, k=200, q=1)

Note the parameter is optional and its default value is zero.
 
## References

[1]: Lyubimov and Palumbo, ["Apache Mahout: Beyond MapReduce; Distributed Algorithm Design"](https://www.amazon.com/Apache-Mahout-MapReduce-Dmitriy-Lyubimov/dp/1523775785)
