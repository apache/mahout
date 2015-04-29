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

import org.apache.mahout.math.drm.DistributedContext
import org.apache.mahout.math.indexeddataset

/**
 * Reader trait is abstract in the sense that the elementReader and rowReader functions must be supplied by an
 * extending trait, which also defines the type to be read.
 * @tparam T type of object to read.
 */
trait Reader[T]{

  val mc: DistributedContext
  val readSchema: Schema

  /**
   * Override in extending trait to supply T and perform a parallel read of collection elements
   * @param mc a [[org.apache.mahout.math.drm.DistributedContext]] to read from
   * @param readSchema map of parameters controlling formating and how the read is executed
   * @param source list of comma delimited files to read from
   * @param existingRowIDs [[indexeddataset.BiDictionary]] containing row IDs that have already
   *                       been applied to this collection--used to synchronize row IDs between several
   *                       collections
   * @return a new collection of type T
   */
  protected def elementReader(
      mc: DistributedContext,
      readSchema: Schema,
      source: String,
      existingRowIDs: Option[BiDictionary] = None): T

  /**
   * Override in extending trait to supply T and perform a parallel read of collection rows
   * @param mc a [[org.apache.mahout.math.drm.DistributedContext]] to read from
   * @param readSchema map of parameters controlling formating and how the read is executed
   * @param source list of comma delimited files to read from
   * @param existingRowIDs [[indexeddataset.BiDictionary]] containing row IDs that have already
   *                       been applied to this collection--used to synchronize row IDs between several
   *                       collections
   * @return a new collection of type T
   */
  protected def rowReader(
      mc: DistributedContext,
      readSchema: Schema,
      source: String,
      existingRowIDs: Option[BiDictionary] = None): T

  /**
   * Public method called to perform the element-wise read. Usually no need to override
   * @param source comma delimited URIs to read from
   * @param existingRowIDs a [[indexeddataset.BiDictionary]] containing previously used id mappings--used
   *                       to synchronize all row ids is several collections
   * @return a new collection of type T
   */
  def readElementsFrom(
      source: String,
      existingRowIDs: Option[BiDictionary] = None): T =
    elementReader(mc, readSchema, source, existingRowIDs)

  /**
   * Public method called to perform the row-wise read. Usually no need to override.
   * @param source comma delimited URIs to read from
   * @param existingRowIDs a [[indexeddataset.BiDictionary]] containing previously used id mappings--used
   *                       to synchronize all row ids is several collections
   * @return  a new collection of type T
   */
  def readRowsFrom(
      source: String,
      existingRowIDs: Option[BiDictionary] = None): T =
    rowReader(mc, readSchema, source, existingRowIDs)
}

/**
 * Writer trait is abstract in the sense that the writer method must be supplied by an extending trait,
 * which also defines the type to be written.
 * @tparam T type of object to write, usually a matrix type thing.
 */
trait Writer[T]{

  val mc: DistributedContext
  val sort: Boolean
  val writeSchema: Schema

  /**
   * Override to provide writer method
   * @param mc context used to do distributed write
   * @param writeSchema map with params to control format and execution of the write
   * @param dest root directory to write to
   * @param collection usually a matrix like collection to write
   * @param sort flags whether to sort the rows by value descending
   */
  protected def writer(mc: DistributedContext, writeSchema: Schema, dest: String, collection: T, sort: Boolean): Unit

  /**
   * Call this method to perform the write, usually no need to override.
   * @param collection what to write
   * @param dest root directory to write to
   */
  def writeTo(collection: T, dest: String) = writer(mc, writeSchema, dest, collection, sort)
}
