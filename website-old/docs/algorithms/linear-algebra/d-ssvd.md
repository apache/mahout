---
layout: algorithm
title: Distributed Stochastic Singular Value Decomposition
theme:
    name: retro-mahout
---

## Intro

Mahout has a distributed implementation of Stochastic Singular Value Decomposition [1] using the parallelization strategy comprehensively defined in Nathan Halko's dissertation ["Randomized methods for computing low-rank approximations of matrices"](http://amath.colorado.edu/faculty/martinss/Pubs/2012_halko_dissertation.pdf) [2].

## Modified SSVD Algorithm

Given an `\(m\times n\)`
matrix `\(\mathbf{A}\)`, a target rank `\(k\in\mathbb{N}_{1}\)`
, an oversampling parameter `\(p\in\mathbb{N}_{1}\)`, 
and the number of additional power iterations `\(q\in\mathbb{N}_{0}\)`, 
this procedure computes an `\(m\times\left(k+p\right)\)`
SVD `\(\mathbf{A\approx U}\boldsymbol{\Sigma}\mathbf{V}^{\top}\)`:

  1. Create seed for random `\(n\times\left(k+p\right)\)`
  matrix `\(\boldsymbol{\Omega}\)`. The seed defines matrix `\(\mathbf{\Omega}\)`
  using Gaussian unit vectors per one of suggestions in [Halko, Martinsson, Tropp].

  2. `\(\mathbf{Y=A\boldsymbol{\Omega}},\,\mathbf{Y}\in\mathbb{R}^{m\times\left(k+p\right)}\)`
 
  3. Column-orthonormalize `\(\mathbf{Y}\rightarrow\mathbf{Q}\)`
  by computing thin decomposition `\(\mathbf{Y}=\mathbf{Q}\mathbf{R}\)`.
  Also, `\(\mathbf{Q}\in\mathbb{R}^{m\times\left(k+p\right)},\,\mathbf{R}\in\mathbb{R}^{\left(k+p\right)\times\left(k+p\right)}\)`; denoted as `\(\mathbf{Q}=\mbox{qr}\left(\mathbf{Y}\right).\mathbf{Q}\)`

  4. `\(\mathbf{B}_{0}=\mathbf{Q}^{\top}\mathbf{A}:\,\,\mathbf{B}\in\mathbb{R}^{\left(k+p\right)\times n}\)`.
 
  5. If `\(q>0\)`
  repeat: for `\(i=1..q\)`: 
  `\(\mathbf{B}_{i}^{\top}=\mathbf{A}^{\top}\mbox{qr}\left(\mathbf{A}\mathbf{B}_{i-1}^{\top}\right).\mathbf{Q}\)`
  (power iterations step).

  6. Compute Eigensolution of a small Hermitian `\(\mathbf{B}_{q}\mathbf{B}_{q}^{\top}=\mathbf{\hat{U}}\boldsymbol{\Lambda}\mathbf{\hat{U}}^{\top}\)`,
  `\(\mathbf{B}_{q}\mathbf{B}_{q}^{\top}\in\mathbb{R}^{\left(k+p\right)\times\left(k+p\right)}\)`.
 
  7. Singular values `\(\mathbf{\boldsymbol{\Sigma}}=\boldsymbol{\Lambda}^{0.5}\)`,
  or, in other words, `\(s_{i}=\sqrt{\sigma_{i}}\)`.
 
  8. If needed, compute `\(\mathbf{U}=\mathbf{Q}\hat{\mathbf{U}}\)`.

  9. If needed, compute `\(\mathbf{V}=\mathbf{B}_{q}^{\top}\hat{\mathbf{U}}\boldsymbol{\Sigma}^{-1}\)`.
Another way is `\(\mathbf{V}=\mathbf{A}^{\top}\mathbf{U}\boldsymbol{\Sigma}^{-1}\)`.




## Implementation

Mahout `dssvd(...)` is implemented in the mahout `math-scala` algebraic optimizer which translates Mahout's R-like linear algebra operators into a physical plan for both Spark and H2O distributed engines.

    def dssvd[K: ClassTag](drmA: DrmLike[K], k: Int, p: Int = 15, q: Int = 0):
        (DrmLike[K], DrmLike[Int], Vector) = {

        val drmAcp = drmA.checkpoint()

        val m = drmAcp.nrow
        val n = drmAcp.ncol
        assert(k <= (m min n), "k cannot be greater than smaller of m, n.")
        val pfxed = safeToNonNegInt((m min n) - k min p)

        // Actual decomposition rank
        val r = k + pfxed

        // We represent Omega by its seed.
        val omegaSeed = RandomUtils.getRandom().nextInt()

        // Compute Y = A*Omega.  
        var drmY = drmAcp.mapBlock(ncol = r) {
            case (keys, blockA) =>
                val blockY = blockA %*% Matrices.symmetricUniformView(n, r, omegaSeed)
            keys -> blockY
        }

        var drmQ = dqrThin(drmY.checkpoint())._1

        // Checkpoint Q if last iteration
        if (q == 0) drmQ = drmQ.checkpoint()

        var drmBt = drmAcp.t %*% drmQ
        
        // Checkpoint B' if last iteration
        if (q == 0) drmBt = drmBt.checkpoint()

        for (i <- 0  until q) {
            drmY = drmAcp %*% drmBt
            drmQ = dqrThin(drmY.checkpoint())._1            
            
            // Checkpoint Q if last iteration
            if (i == q - 1) drmQ = drmQ.checkpoint()
            
            drmBt = drmAcp.t %*% drmQ
            
            // Checkpoint B' if last iteration
            if (i == q - 1) drmBt = drmBt.checkpoint()
        }

        val (inCoreUHat, d) = eigen(drmBt.t %*% drmBt)
        val s = d.sqrt

        // Since neither drmU nor drmV are actually computed until actually used
        // we don't need the flags instructing compute (or not compute) either of the U,V outputs 
        val drmU = drmQ %*% inCoreUHat
        val drmV = drmBt %*% (inCoreUHat %*%: diagv(1 /: s))

        (drmU(::, 0 until k), drmV(::, 0 until k), s(0 until k))
    }

Note: As a side effect of checkpointing, U and V values are returned as logical operators (i.e. they are neither checkpointed nor computed).  Therefore there is no physical work actually done to compute `\(\mathbf{U}\)` or `\(\mathbf{V}\)` until they are used in a subsequent expression.


## Usage

The scala `dssvd(...)` method can easily be called in any Spark or H2O application built with the `math-scala` library and the corresponding `Spark` or `H2O` engine module as follows:

    import org.apache.mahout.math._
    import decompositions._
    import drm._
    
    
    val(drmU, drmV, s) = dssvd(drma, k = 40, q = 1)

 
## References

[1]: [Mahout Scala and Mahout Spark Bindings for Linear Algebra Subroutines](http://mahout.apache.org/users/sparkbindings/ScalaSparkBindings.pdf)

[2]: [Randomized methods for computing low-rank
approximations of matrices](http://amath.colorado.edu/faculty/martinss/Pubs/2012_halko_dissertation.pdf)

[2]: [Halko, Martinsson, Tropp](http://arxiv.org/abs/0909.4061)

[3]: [Mahout Spark and Scala Bindings](http://mahout.apache.org/users/sparkbindings/home.html)



