---
layout: default
title: Distributed Cholesky QR
theme:
    name: retro-mahout
---

# Distributed Cholesky QR


## Intro

Mahout has a distributed implementation of QR decomposition for tall thin matrices[1].

## Algorithm 

For the classic QR decomposition of the form `\(\mathbf{A}=\mathbf{QR},\mathbf{A}\in\mathbb{R}^{m\times n}\)` a distributed version is fairly easily achieved if `\(\mathbf{A}\)` is tall and thin such that `\(\mathbf{A}^{\top}\mathbf{A}\)` fits in memory, i.e. *m* is large but *n* < ~5000 Under such circumstances, only `\(\mathbf{A}\)` and `\(\mathbf{Q}\)` are distributed matrices and `\(\mathbf{A^{\top}A}\)` and `\(\mathbf{R}\)` are in-core products. We just compute the in-core version of the Cholesky decomposition in the form of `\(\mathbf{LL}^{\top}= \mathbf{A}^{\top}\mathbf{A}\)`.  After that we take `\(\mathbf{R}= \mathbf{L}^{\top}\)` and `\(\mathbf{Q}=\mathbf{A}\left(\mathbf{L}^{\top}\right)^{-1}\)`.  The latter is easily achieved by multiplying each vertical block of `\(\mathbf{A}\)` by `\(\left(\mathbf{L}^{\top}\right)^{-1}\)`.  (There is no actual matrix inversion happening). 



## Implementation

Mahout `dqrThin(...)` is implemented in the mahout `math-scala` algebraic optimizer which translates Mahout's R-like linear algebra operators into a physical plan for both Spark and H2O distributed engines.

    def dqrThin[K: ClassTag](A: DrmLike[K], checkRankDeficiency: Boolean = true): (DrmLike[K], Matrix) = {        
        if (drmA.ncol > 5000)
            log.warn("A is too fat. A'A must fit in memory and easily broadcasted.")
        implicit val ctx = drmA.context
        val AtA = (drmA.t %*% drmA).checkpoint()
        val inCoreAtA = AtA.collect
        val ch = chol(inCoreAtA)
        val inCoreR = (ch.getL cloned) t
        if (checkRankDeficiency && !ch.isPositiveDefinite)
            throw new IllegalArgumentException("R is rank-deficient.")
        val bcastAtA = sc.broadcast(inCoreAtA)
        val Q = A.mapBlock() {
            case (keys, block) => keys -> chol(bcastAtA).solveRight(block)
        }
        Q -> inCoreR
    }


## Usage

The scala `dqrThin(...)` method can easily be called in any Spark or H2O application built with the `math-scala` library and the corresponding `Spark` or `H2O` engine module as follows:

    import org.apache.mahout.math._
    import decompositions._
    import drm._
    
    val(drmQ, inCoreR) = dqrThin(drma)

 
## References

[1]: [Mahout Scala and Mahout Spark Bindings for Linear Algebra Subroutines](http://mahout.apache.org/users/sparkbindings/ScalaSparkBindings.pdf)

[2]: [Mahout Spark and Scala Bindings](http://mahout.apache.org/users/sparkbindings/home.html)

