package org.rawkintrevo.cylon.common.mahout

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.nio.ByteBuffer

import org.apache.mahout.math._
import org.apache.mahout.math.algorithms.preprocessing.MeanCenter
import org.apache.mahout.math.decompositions._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings.MahoutCollections._

import scala.collection.JavaConversions._

object MahoutUtils {

  def matrixWriter(m: Matrix, path: String): Unit = {
    val oos = new ObjectOutputStream(new FileOutputStream(path))
    oos.writeObject(m.toArray.map(v => v.toArray).toList)
    oos.close
  }

  def matrixReader(path: String): Matrix ={
    val ois = new ObjectInputStream(new FileInputStream(path))
    // for row in matrix
    val la = ois.readObject.asInstanceOf[List[Array[Double]]]
    ois.close
    val m = listArrayToMatrix(la)
    m
  }

  def vectorReader(path: String): Vector ={
    val m = matrixReader(path)
    m(0, ::)
  }

  def listArrayToMatrix(la: List[Array[Double]]): Matrix = {
    dense(la.map(m => dvec(m)):_*)
  }

  def decomposeImgVecWithEigenfaces(v: Vector, m: Matrix): Vector = {

    val XtX = m.t %*% m
    val Xty = m.t %*% v
    solve(XtX, Xty).viewPart(3, m.numCols()-3)  // The first 3 eigenfaces often only capture 3 dimensional light, which we want to ignore

  }

  def vector2byteArray(v: Vector): Array[Byte] = {
    val bb: ByteBuffer = ByteBuffer.allocate(v.size() * 8)
    for (d <- v.toArray){
      bb.putDouble(d)
    }
    bb.array()
  }

  def byteArray2vector(ba: Array[Byte]): Vector = {
    val bb: ByteBuffer = ByteBuffer.wrap(ba)
    val output: Array[Double] = new Array[Double](ba.length / 8)
    for (i <- output.indices) {
      output(i) = bb.getDouble()
    }
    dvec(output)
  }
}
