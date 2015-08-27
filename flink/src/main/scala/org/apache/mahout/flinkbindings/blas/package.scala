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

import scala.reflect.ClassTag
import org.apache.flink.api.java.functions.KeySelector
import org.apache.flink.api.java.typeutils.ResultTypeQueryable
import org.apache.flink.api.scala.createTypeInformation
import org.apache.flink.api.common.typeinfo.TypeInformation

package object blas {

  // TODO: remove it once figure out how to make Flink accept interfaces (Vector here)
  def selector[V, K: ClassTag]: KeySelector[(K, V), K] = {
    val tag = implicitly[ClassTag[K]]
    if (tag.runtimeClass.equals(classOf[Int])) {
      tuple_1_int.asInstanceOf[KeySelector[(K, V), K]]
    } else if (tag.runtimeClass.equals(classOf[Long])) {
      tuple_1_long.asInstanceOf[KeySelector[(K, V), K]]
    } else if (tag.runtimeClass.equals(classOf[String])) {
      tuple_1_string.asInstanceOf[KeySelector[(K, V), K]]
    } else {
      throw new IllegalArgumentException(s"index type $tag is not supported")
    }
  }

  private def tuple_1_int[K: ClassTag] = new KeySelector[(Int, _), Int] 
                  with ResultTypeQueryable[Int] {
    def getKey(tuple: Tuple2[Int, _]): Int = tuple._1
    def getProducedType: TypeInformation[Int] = createTypeInformation[Int]
  }

  private def tuple_1_long[K: ClassTag] = new KeySelector[(Long, _), Long] 
                  with ResultTypeQueryable[Long] {
    def getKey(tuple: Tuple2[Long, _]): Long = tuple._1
    def getProducedType: TypeInformation[Long] = createTypeInformation[Long]
  }

  private def tuple_1_string[K: ClassTag] = new KeySelector[(String, _), String] 
                  with ResultTypeQueryable[String] {
    def getKey(tuple: Tuple2[String, _]): String = tuple._1
    def getProducedType: TypeInformation[String] = createTypeInformation[String]
  }
}