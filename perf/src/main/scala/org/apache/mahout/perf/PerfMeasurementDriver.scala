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

package org.apache.mahout.perf
import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math.{Matrices, SparseRowMatrix, scalabindings}
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.decompositions._
import java.io.{BufferedWriter, FileWriter}
import com.github.tototoshi.csv.CSVWriter
import RLikeOps._



object PerfMeasurementDriver extends App {



  val out = new BufferedWriter(new FileWriter("mahoutRuntimePerf.csv"));
  val writer = new CSVWriter(out);
  val t1=time{doSSVD}
  //println("t1="+t1)
  val t2=time{doSPCA}
  //println("t2="+t2)
  var listOfRecords= Array(t1.toString,t2.toString)
  writer.writeRow(listOfRecords)
  out.close()


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

    //printf("Svs-control:%s\n", sControl)
    pcaControl = (pcaControl %*%: diagv(sControl))(::, 0 until k)

    //printf("pca:\n%s\n", pca(0 until 10, 0 until 10))
    //printf("pcaControl:\n%s\n", pcaControl(0 until 10, 0 until 10))
  }





  def time[R](block: => R): Long = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    t1-t0
  }

}