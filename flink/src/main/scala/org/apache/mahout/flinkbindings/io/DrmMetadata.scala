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
package org.apache.mahout.flinkbindings.io

import scala.reflect.ClassTag
import org.apache.hadoop.io._
import java.util.Arrays

/**
 * Flink DRM Metadata
 */
class DrmMetadata(

  /** Writable  key type as a sub-type of Writable */
  val keyTypeWritable: Class[_],

  /** Value writable type, as a sub-type of Writable */
  val valueTypeWritable: Class[_]) {

  import DrmMetadata._

  /**
   * @param keyClassTag: Actual drm key class tag once converted out of writable
   * @param keyW2ValFunc: Conversion from Writable to value type of the DRM key
   */
  val (keyClassTag: ClassTag[_], unwrapKeyFunction: ((Writable) => Any)) = keyTypeWritable match {
    case cz if cz == classOf[IntWritable] => ClassTag.Int -> w2int _
    case cz if cz == classOf[LongWritable] => ClassTag.Long -> w2long _
    case cz if cz == classOf[DoubleWritable] => ClassTag.Double -> w2double _
    case cz if cz == classOf[FloatWritable] => ClassTag.Float -> w2float _
    case cz if cz == classOf[Text] => ClassTag(classOf[String]) -> w2string _
    case cz if cz == classOf[BooleanWritable] => ClassTag(classOf[Boolean]) -> w2bool _
    case cz if cz == classOf[BytesWritable] => ClassTag(classOf[Array[Byte]]) -> w2bytes _
    case _ => throw new IllegalArgumentException(s"Unsupported DRM key type:${keyTypeWritable.getName}")
  }

}

object DrmMetadata {

  private[io] def w2int(w: Writable) = w.asInstanceOf[IntWritable].get()

  private[io] def w2long(w: Writable) = w.asInstanceOf[LongWritable].get()

  private[io] def w2double(w: Writable) = w.asInstanceOf[DoubleWritable].get()

  private[io] def w2float(w: Writable) = w.asInstanceOf[FloatWritable].get()

  private[io] def w2string(w: Writable) = w.asInstanceOf[Text].toString()

  private[io] def w2bool(w: Writable) = w.asInstanceOf[BooleanWritable].get()

  private[io] def w2bytes(w: Writable) = Arrays.copyOf(w.asInstanceOf[BytesWritable].getBytes(),
    w.asInstanceOf[BytesWritable].getLength())
}
