/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.mahout.flinkbindings

import org.apache.mahout.math.Matrix
import org.apache.mahout.math.MatrixWritable
import org.apache.mahout.math.Vector
import org.apache.mahout.math.VectorWritable
import org.apache.mahout.math.drm.BCast

import com.google.common.io.ByteStreams

/**
 * FlinkByteBCast wraps vector/matrix objects, represents them as byte arrays, and when 
 * it's used in UDFs, they are serialized using standard Java serialization along with 
 * UDFs (as a part of closure) and broadcasted to worker nodes. 
 * 
 * There should be a smarter way of doing it with some macro and then rewriting the UDF and 
 * appending `withBroadcastSet` to flink dataSet pipeline, but it's not implemented at the moment.  
 */
class FlinkByteBCast[T](private val arr: Array[Byte]) extends BCast[T] with Serializable {

  private lazy val _value = {
    val stream = ByteStreams.newDataInput(arr)
    val streamType = stream.readInt()

    if (streamType == FlinkByteBCast.StreamTypeVector) {
      val writeable = new VectorWritable()
      writeable.readFields(stream)
    //  printf("broadcastValue: \n%s\n",writeable.get.asInstanceOf[T])
      writeable.get.asInstanceOf[T]
    } else if (streamType == FlinkByteBCast.StreamTypeMatrix) {
      val writeable = new MatrixWritable()
      writeable.readFields(stream)
     // printf("broadcastValue: \n%s\n",writeable.get.asInstanceOf[T])
      writeable.get.asInstanceOf[T]
    } else {
      throw new IllegalArgumentException(s"unexpected type tag $streamType")
    }

  }

  override def value: T = _value

  override def close: Unit = {
    // nothing to close
  }

}

object FlinkByteBCast {

  val StreamTypeVector = 0x0000
  val StreamTypeMatrix = 0xFFFF

  def wrap(v: Vector): FlinkByteBCast[Vector] = {
    val writeable = new VectorWritable(v)
    val dataOutput = ByteStreams.newDataOutput()
    dataOutput.writeInt(StreamTypeVector)
    writeable.write(dataOutput)
    val array = dataOutput.toByteArray()
    new FlinkByteBCast[Vector](array)
  }

  def wrap(m: Matrix): FlinkByteBCast[Matrix] = {
    val writeable = new MatrixWritable(m)
    val dataOutput = ByteStreams.newDataOutput()
    dataOutput.writeInt(StreamTypeMatrix)
    writeable.write(dataOutput)
    val array = dataOutput.toByteArray()
    new FlinkByteBCast[Matrix](array)
  }

}