/**
  * Created by saikat on 6/6/16.
  */
package org.apache.mahout.perf
import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math.{Matrices, SparseRowMatrix, scalabindings}
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.decompositions._
import RLikeOps._



object PerfMeasurementDriver extends App {
  time{doSSVD}
  time{doSPCA}
  def doSSVD() {
    // Very naive, a full-rank only here.
    val a = dense(
      (1, 2, 3),
      (3, 4, 5),
      (-2, 6, 7),
      (-3, 8, 9)
    )

    val rank = 2
    val (u, v, s) = ssvd(a, k = rank, q = 1)

    val (uControl, vControl, sControl) = svd(a)

    printf("U:\n%s\n", u)
    printf("U-control:\n%s\n", uControl)
    printf("V:\n%s\n", v)
    printf("V-control:\n%s\n", vControl)
    printf("Sigma:\n%s\n", s)
    printf("Sigma-control:\n%s\n", sControl)

  }

  def doSPCA: Unit = {
    import math._

    val rnd = RandomUtils.getRandom

    // Number of points
    val m = 500
    // Length of actual spectrum
    val spectrumLen = 40

    val spectrum = dvec((0 until spectrumLen).map(x => 300.0 * exp(-x) max 1e-3))
    printf("spectrum:%s\n", spectrum)

    val (u, _) = qr(new SparseRowMatrix(m, spectrumLen) :=
      ((r, c, v) => if (rnd.nextDouble() < 0.2) 0 else rnd.nextDouble() + 5.0))

    // PCA Rotation matrix -- should also be orthonormal.
    val (tr, _) = qr(Matrices.symmetricUniformView(spectrumLen, spectrumLen, rnd.nextInt) - 10.0)

    val input = (u %*%: diagv(spectrum)) %*% tr.t

    // Calculate just first 10 principal factors and reduce dimensionality.
    // Since we assert just validity of the s-pca, not stochastic error, we bump p parameter to
    // ensure to zero stochastic error and assert only functional correctness of the method's pca-
    // specific additions.
    val k = 10
    var (pca, _, s) = spca(a = input, k = k, p = spectrumLen, q = 1)
    printf("Svs:%s\n", s)
    // Un-normalized pca data:
    pca = pca %*%: diagv(s)

    // Of course, once we calculated the pca, the spectrum is going to be different since our originally
    // generated input was not centered. So here, we'd just brute-solve pca to verify
    val xi = input.colMeans()
    for (r <- 0 until input.nrow) input(r, ::) -= xi
    var (pcaControl, _, sControl) = svd(m = input)

    printf("Svs-control:%s\n", sControl)
    pcaControl = (pcaControl %*%: diagv(sControl))(::, 0 until k)

    printf("pca:\n%s\n", pca(0 until 10, 0 until 10))
    printf("pcaControl:\n%s\n", pcaControl(0 until 10, 0 until 10))
  }





  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
    result
  }

}