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

package org.apache.mahout.sparkbindings.io

import com.esotericsoftware.kryo.Kryo
import org.apache.mahout.common.io.{VectorKryoSerializer, GenericMatrixKryoSerializer}
import org.apache.mahout.math._
import org.apache.spark.serializer.KryoRegistrator
import org.apache.mahout.logging._

object MahoutKryoRegistrator {

  private final implicit val log = getLog(this.getClass)

  def registerClasses(kryo: Kryo) = {

    trace("Registering mahout classes.")

    kryo.register(classOf[SparseColumnMatrix], new UnsupportedSerializer)
    kryo.addDefaultSerializer(classOf[Vector], new VectorKryoSerializer())
    kryo.addDefaultSerializer(classOf[Matrix], new GenericMatrixKryoSerializer)

    Seq(
      classOf[Vector],
      classOf[Matrix],
      classOf[DiagonalMatrix],
      classOf[DenseMatrix],
      classOf[SparseRowMatrix],
      classOf[SparseMatrix],
      classOf[MatrixView],
      classOf[MatrixSlice],
      classOf[TransposedMatrixView],
      classOf[DenseVector],
      classOf[RandomAccessSparseVector],
      classOf[SequentialAccessSparseVector],
      classOf[MatrixVectorView],
      classOf[VectorView],
      classOf[PermutedVectorView],
      classOf[Array[Vector]],
      classOf[Array[Matrix]],
      Class.forName(classOf[DiagonalMatrix].getName + "$SingleElementVector"),
      Class.forName(classOf[DenseVector].getName + "$DenseVectorView"),
      // This is supported by twitter.chill, but kryo still is offended by lack of registration:
      classOf[Range],
      //classOf[Unit], // this causes an error with "void not serializable" or some such on a real cluster. Not found
      // in unit tests
      classOf[scala.collection.mutable.WrappedArray.ofRef[_]],
      classOf[Array[Int]],
      classOf[Array[String]]

    ) foreach kryo.register

  }

}

/** Kryo serialization registrator for Mahout */
class MahoutKryoRegistrator extends KryoRegistrator {

  override def registerClasses(kryo: Kryo) = MahoutKryoRegistrator.registerClasses(kryo)
}
