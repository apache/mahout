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

package org.apache.mahout.common.io

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, Serializer}
import org.apache.log4j.Logger
import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.flavor.TraversingStructureEnum
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

import scala.collection.JavaConversions._

object GenericMatrixKryoSerializer {

  private implicit final val log = Logger.getLogger(classOf[GenericMatrixKryoSerializer])

}

/** Serializes Sparse or Dense in-core generic matrix (row-wise or column-wise backed) */
class GenericMatrixKryoSerializer extends Serializer[Matrix] {

  import GenericMatrixKryoSerializer._

  override def write(kryo: Kryo, output: Output, mx: Matrix): Unit = {

    debug(s"Writing mx of type ${mx.getClass.getName}")

    val structure = mx.getFlavor.getStructure

    // Write structure bit
    output.writeInt(structure.ordinal(), true)

    // Write geometry
    output.writeInt(mx.nrow, true)
    output.writeInt(mx.ncol, true)

    // Write in most efficient traversal order (using backing vectors perhaps)
    structure match {
      case TraversingStructureEnum.COLWISE => writeRowWise(kryo, output, mx.t)
      case TraversingStructureEnum.SPARSECOLWISE => writeSparseRowWise(kryo, output, mx.t)
      case TraversingStructureEnum.SPARSEROWWISE => writeSparseRowWise(kryo, output, mx)
      case TraversingStructureEnum.VECTORBACKED => writeVectorBacked(kryo, output, mx)
      case _ => writeRowWise(kryo, output, mx)
    }

  }

  private def writeVectorBacked(kryo: Kryo, output: Output, mx: Matrix) {

    require(mx != null)

    // At this point we are just doing some vector-backed classes individually. TODO: create
    // api to obtain vector-backed matrix data.
    kryo.writeClass(output, mx.getClass)
    mx match {
      case mxD: DiagonalMatrix => kryo.writeObject(output, mxD.diagv)
      case mxS: DenseSymmetricMatrix => kryo.writeObject(output, dvec(mxS.getData))
      case mxT: UpperTriangular => kryo.writeObject(output, dvec(mxT.getData))
      case _ => throw new IllegalArgumentException(s"Unsupported matrix type:${mx.getClass.getName}")
    }
  }

  private def readVectorBacked(kryo: Kryo, input: Input, nrow: Int, ncol: Int) = {

    // We require vector-backed matrices to have vector-parameterized constructor to construct.
    val clazz = kryo.readClass(input).getType

    debug(s"Deserializing vector-backed mx of type ${clazz.getName}.")

    clazz.getConstructor(classOf[Vector]).newInstance(kryo.readObject(input, classOf[Vector])).asInstanceOf[Matrix]
  }

  private def writeRowWise(kryo: Kryo, output: Output, mx: Matrix): Unit = {
    for (row <- mx) kryo.writeObject(output, row)
  }

  private def readRows(kryo: Kryo, input: Input, nrow: Int) = {
    Array.tabulate(nrow) { _ => kryo.readObject(input, classOf[Vector])}
  }

  private def readSparseRows(kryo: Kryo, input: Input) = {

    // Number of slices
    val nslices = input.readInt(true)

    Array.tabulate(nslices) { _ =>
      input.readInt(true) -> kryo.readObject(input, classOf[Vector])
    }
  }

  private def writeSparseRowWise(kryo: Kryo, output: Output, mx: Matrix): Unit = {

    val nslices = mx.numSlices()

    output.writeInt(nslices, true)

    var actualNSlices = 0
    for (row <- mx.iterateNonEmpty()) {
      output.writeInt(row.index(), true)
      kryo.writeObject(output, row.vector())
      actualNSlices += 1
    }

    require(nslices == actualNSlices, "Number of slices reported by Matrix.numSlices() was different from actual " +
      "slice iterator size.")
  }

  override def read(kryo: Kryo, input: Input, mxClass: Class[Matrix]): Matrix = {

    // Read structure hint
    val structure = TraversingStructureEnum.values()(input.readInt(true))

    // Read geometry
    val nrow = input.readInt(true)
    val ncol = input.readInt(true)

    debug(s"read matrix geometry: $nrow x $ncol.")

    structure match {

      // Sparse or dense column wise
      case TraversingStructureEnum.COLWISE =>
        val cols = readRows(kryo, input, ncol)

        if (!cols.isEmpty && cols.head.isDense)
          dense(cols).t
        else {
          debug("Deserializing as SparseRowMatrix.t (COLWISE).")
          new SparseRowMatrix(ncol, nrow, cols, true, false).t
        }

      // transposed SparseMatrix case
      case TraversingStructureEnum.SPARSECOLWISE =>
        val cols = readSparseRows(kryo, input)
        val javamap = new java.util.HashMap[Integer, Vector]((cols.size << 1) + 1)
        cols.foreach { case (idx, vec) => javamap.put(idx, vec)}

        debug("Deserializing as SparseMatrix.t (SPARSECOLWISE).")
        new SparseMatrix(ncol, nrow, javamap, true).t

      // Sparse Row-wise -- this will be created as a SparseMatrix.
      case TraversingStructureEnum.SPARSEROWWISE =>
        val rows = readSparseRows(kryo, input)
        val javamap = new java.util.HashMap[Integer, Vector]((rows.size << 1) + 1)
        rows.foreach { case (idx, vec) => javamap.put(idx, vec)}

        debug("Deserializing as SparseMatrix (SPARSEROWWISE).")
        new SparseMatrix(nrow, ncol, javamap, true)
      case TraversingStructureEnum.VECTORBACKED =>

        debug("Deserializing vector-backed...")
        readVectorBacked(kryo, input, nrow, ncol)

      // By default, read row-wise.
      case _ =>
        val cols = readRows(kryo, input, nrow)
        // this still copies a lot of stuff...
        if (!cols.isEmpty && cols.head.isDense) {

          debug("Deserializing as DenseMatrix.")
          dense(cols)
        } else {

          debug("Deserializing as SparseRowMatrix(default).")
          new SparseRowMatrix(nrow, ncol, cols, true, false)
        }
    }

  }
}
