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

import org.apache.flink.api.java.functions.KeySelector
import org.apache.mahout.math.Vector
import scala.reflect.ClassTag

package object blas {

  // TODO: remove it once figure out how to make Flink accept interfaces (Vector here)
  def tuple_1[K: ClassTag] = new KeySelector[(Int, K), Integer] {
    def getKey(tuple: Tuple2[Int, K]): Integer = tuple._1
  }

}