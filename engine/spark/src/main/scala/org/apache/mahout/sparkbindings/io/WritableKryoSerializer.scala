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

import com.esotericsoftware.kryo.{Kryo, Serializer}
import com.esotericsoftware.kryo.io.{Input, Output}
import org.apache.hadoop.io.{DataInputBuffer, DataOutputBuffer, Writable}
import scala.reflect.ClassTag

class WritableKryoSerializer[V <% Writable, W <: Writable <% V : ClassTag] extends Serializer[V] {

  def write(kryo: Kryo, out: Output, v: V) = {
    val dob = new DataOutputBuffer()
    v.write(dob)
    dob.close()

    out.writeInt(dob.getLength)
    out.write(dob.getData, 0, dob.getLength)
  }

  def read(kryo: Kryo, in: Input, vClazz: Class[V]): V = {
    val dib = new DataInputBuffer()
    val len = in.readInt()
    val data = new Array[Byte](len)
    in.read(data)
    dib.reset(data, len)
    val w: W = implicitly[ClassTag[W]].runtimeClass.newInstance().asInstanceOf[W]
    w.readFields(dib)
    w

  }
}
