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

package org.apache.mahout.math.indexeddataset

import com.google.common.collect.{BiMap, HashBiMap}
import org.apache.mahout.math.drm.DistributedContext

/** Reader trait is abstract in the sense that the elementReader function must be defined by an extending trait,
  * which also defines the type to be read.
  * @tparam T type of object to read.
  */
trait Reader[T]{

  val mc: DistributedContext
  val readSchema: Schema

  protected def elementReader(
      mc: DistributedContext,
      readSchema: Schema,
      source: String,
      existingRowIDs: BiMap[String, Int]): T

  protected def drmReader(
      mc: DistributedContext,
      readSchema: Schema,
      source: String,
      existingRowIDs: BiMap[String, Int]): T

  def readElementsFrom(
      source: String,
      existingRowIDs: BiMap[String, Int] = HashBiMap.create()): T =
    elementReader(mc, readSchema, source, existingRowIDs)

  def readDRMFrom(
      source: String,
      existingRowIDs: BiMap[String, Int] = HashBiMap.create()): T =
    drmReader(mc, readSchema, source, existingRowIDs)
}

/** Writer trait is abstract in the sense that the writer method must be supplied by an extending trait,
  * which also defines the type to be written.
  * @tparam T type of object to write.
  */
trait Writer[T]{

  val mc: DistributedContext
  val sort: Boolean
  val writeSchema: Schema

  protected def writer(mc: DistributedContext, writeSchema: Schema, dest: String, collection: T, sort: Boolean): Unit

  def writeTo(collection: T, dest: String) = writer(mc, writeSchema, dest, collection, sort)
}
