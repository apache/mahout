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

import scala.collection.immutable.HashMap

/**
 * Immutable Bi-directional Map.
 * @param m Map to use for forward reference
 * @param i optional reverse map of value to key, will create one lazily if none is provided
 *          and is required to have no duplicate reverse mappings.
 */
class BiMap[K, V] (
    private val m: Map[K, V],
    // if this is serialized we allow i to be discarded and recalculated when deserialized
    @transient private var i: Option[BiMap[V, K]] = None
  ) extends Serializable {

  // NOTE: make inverse's inverse point back to current BiMap
  // if this is serialized we allow inverse to be discarded and recalculated when deserialized
  @transient lazy val inverse: BiMap[V, K] = {
    if( i == null.asInstanceOf[Option[BiMap[V, K]]] )
      i = None
    i.getOrElse {
      val rev = m.map(_.swap)
      require((rev.size == m.size), "Failed to create reversed map. Cannot have duplicated values.")
      new BiMap(rev, Some(this))
    }
  }

  // forces inverse to be calculated in the constructor when deserialized
  // not when first used
  @transient val size_ = inverse.size

  def get(k: K): Option[V] = m.get(k)

  def getOrElse(k: K, default: => V): V = m.getOrElse(k, default)

  def contains(k: K): Boolean = m.contains(k)

  def apply(k: K): V = m.apply(k)

  /**
   * Converts to a map.
   * @return a map of type immutable.Map[K, V]
   */
  def toMap: Map[K, V] = m

  /**
   * Converts to a sequence.
   * @return a sequence containing all elements of this map
   */
  def toSeq: Seq[(K, V)] = m.toSeq

  def size: Int = m.size

  def take(n: Int) = BiMap(m.take(n))

  override def toString = m.toString
}

object BiMap {

  /** Extra constructor from a map */
  def apply[K, V](x: Map[K, V]): BiMap[K, V] = new BiMap(x)

}

/** BiDictionary is a specialized BiMap that has non-negative Ints as values for use as DRM keys */
class BiDictionary (
    private val m: Map[String, Int],
    @transient private val i: Option[BiMap[Int, String]] = None )
  extends BiMap[String, Int](m, i) {

  /**
   * Create a new BiDictionary with the keys supplied and values ranging from 0 to size -1
   * @param keys a set of String
   */
  def this(keys: Seq[String]) = {
    this(HashMap(keys.view.zipWithIndex: _*))
  }

  def merge(
    keys: Seq[String]): BiDictionary = {

    var newIDs = List[String]()

    for (key <- keys) {
      if (!m.contains(key)) newIDs = key +: newIDs
    }
    if(newIDs.isEmpty) this else new BiDictionary(m ++ HashMap(newIDs.view.zip (Stream from size): _*))

  }

}

/** BiDictionary is a specialized BiMap that has non-negative Ints as values for use as DRM keys.
  * The companion object provides modification methods specific to maintaining contiguous Int values
  * and unique String keys */
object BiDictionary {

  /**
   * Append new keys to an existing BiDictionary and return the result. The values will start
   * at m.size and increase to create a continuous non-zero value set from 0 to size - 1
   * @param keys new keys to append, not checked for uniqueness so may be dangerous
   * @param biDi merge keys to this BiDictionary and create new values buy incremeting from the highest Int value
   * @return a BiDictionary with added mappings
   */
  /*def append(keys: Seq[String], biDi: BiDictionary): BiDictionary = {
    val hm = HashMap(keys.view.zip (Stream from biDi.size): _*)
    new BiDictionary(biDi.m ++ hm)
  }*/

}
