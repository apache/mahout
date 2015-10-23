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
import org.apache.mahout.logging._
import org.apache.mahout.math._
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.collection.JavaConversions._


object VectorKryoSerializer {

  final val FLAG_DENSE: Int = 0x01
  final val FLAG_SEQUENTIAL: Int = 0x02
  final val FLAG_NAMED: Int = 0x04
  final val FLAG_LAX_PRECISION: Int = 0x08

  private final implicit val log = getLog(classOf[VectorKryoSerializer])

}

class VectorKryoSerializer(val laxPrecision: Boolean = false) extends Serializer[Vector] {

  import VectorKryoSerializer._

  override def write(kryo: Kryo, output: Output, vector: Vector): Unit = {

    require(vector != null)

    trace(s"Serializing vector of ${vector.getClass.getName} class.")

    // Write length
    val len = vector.length
    output.writeInt(len, true)

    // Interrogate vec properties
    val dense = vector.isDense
    val sequential = vector.isSequentialAccess
    val named = vector.isInstanceOf[NamedVector]

    var flag = 0

    if (dense) {
      flag |= FLAG_DENSE
    } else if (sequential) {
      flag |= FLAG_SEQUENTIAL
    }

    if (vector.isInstanceOf[NamedVector]) {
      flag |= FLAG_NAMED
    }

    if (laxPrecision) flag |= FLAG_LAX_PRECISION

    // Write flags
    output.writeByte(flag)

    // Write name if needed
    if (named) output.writeString(vector.asInstanceOf[NamedVector].getName)

    dense match {

      // Dense vector.
      case true =>

        laxPrecision match {
          case true =>
            for (i <- 0 until vector.length) output.writeFloat(vector(i).toFloat)
          case _ =>
            for (i <- 0 until vector.length) output.writeDouble(vector(i))
        }
      case _ =>

        // Turns out getNumNonZeroElements must check every element if it is indeed non-zero. The
        // iterateNonZeros() on the other hand doesn't do that, so that's all inconsistent right
        // now. so we'll just auto-terminate.
        val iter = vector.nonZeroes.toIterator.filter(_.get() != 0.0)

        sequential match {

          // Delta encoding
          case true =>

            var idx = 0
            laxPrecision match {
              case true =>
                while (iter.hasNext) {
                  val el = iter.next()
                  output.writeFloat(el.toFloat)
                  output.writeInt(el.index() - idx, true)
                  idx = el.index
                }
                // Terminate delta encoding.
                output.writeFloat(0.0.toFloat)
              case _ =>
                while (iter.hasNext) {
                  val el = iter.next()
                  output.writeDouble(el.get())
                  output.writeInt(el.index() - idx, true)
                  idx = el.index
                }
                // Terminate delta encoding.
                output.writeDouble(0.0)
            }

          // Random access.
          case _ =>

            laxPrecision match {

              case true =>
                iter.foreach { el =>
                  output.writeFloat(el.get().toFloat)
                  output.writeInt(el.index(), true)
                }
                // Terminate random access with 0.0 value.
                output.writeFloat(0.0.toFloat)
              case _ =>
                iter.foreach { el =>
                  output.writeDouble(el.get())
                  output.writeInt(el.index(), true)
                }
                // Terminate random access with 0.0 value.
                output.writeDouble(0.0)
            }

        }

    }
  }

  override def read(kryo: Kryo, input: Input, vecClass: Class[Vector]): Vector = {

    val len = input.readInt(true)
    val flags = input.readByte().toInt
    val name = if ((flags & FLAG_NAMED) != 0) Some(input.readString()) else None

    val vec: Vector = flags match {

      // Dense
      case _: Int if (flags & FLAG_DENSE) != 0 =>

        trace(s"Deserializing dense vector.")

        if ((flags & FLAG_LAX_PRECISION) != 0) {
          new DenseVector(len) := { _ => input.readFloat()}
        } else {
          new DenseVector(len) := { _ => input.readDouble()}
        }

      // Sparse case.
      case _ =>

        flags match {

          // Sequential.
          case _: Int if (flags & FLAG_SEQUENTIAL) != 0 =>

            trace("Deserializing as sequential sparse vector.")

            val v = new SequentialAccessSparseVector(len)
            var idx = 0
            var stop = false

            if ((flags & FLAG_LAX_PRECISION) != 0) {

              while (!stop) {
                val value = input.readFloat()
                if (value == 0.0) {
                  stop = true
                } else {
                  idx += input.readInt(true)
                  v(idx) = value
                }
              }
            } else {
              while (!stop) {
                val value = input.readDouble()
                if (value == 0.0) {
                  stop = true
                } else {
                  idx += input.readInt(true)
                  v(idx) = value
                }
              }
            }
            v

          // Random access
          case _ =>

            trace("Deserializing as random access vector.")

            // Read pairs until we see 0.0 value. Prone to corruption attacks obviously.
            val v = new RandomAccessSparseVector(len)
            var stop = false
            if ((flags & FLAG_LAX_PRECISION) != 0) {
              while (! stop ) {
                val value = input.readFloat()
                if ( value == 0.0 ) {
                  stop = true
                } else {
                  val idx = input.readInt(true)
                  v(idx) = value
                }
              }
            } else {
              while (! stop ) {
                val value = input.readDouble()
                if (value == 0.0) {
                  stop = true
                } else {
                  val idx = input.readInt(true)
                  v(idx) = value
                }
              }
            }
            v
        }
    }

    name.map{name =>

      trace(s"Recovering named vector's name $name.")

      new NamedVector(vec, name)
    }
      .getOrElse(vec)
  }
}
