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
 * Immutable Bi-directional dictionary. Expected use is to create an Int or Long value that maps to and from a
 * String key though key and value types are not restricted. Helper functions create sequential Int from
 * 0 to # of Stings - 1. A Hashmap is created for forward and another for inverse mapping. To add more String
 * keys will extend the range of the Int values and append the new (key -> value) mappings.
 * @param m Map to use for forward reference
 * @param i optional reverse map of value to key, will create one in the constructor if none is provided
 *          and is required to have no duplicate reverse mappings.
 */
class BiMap[K, V] (
    private val m: Map[K, V],
    private val i: Option[BiMap[V, K]] = None
  ) extends Serializable {

  // NOTE: make inverse's inverse point back to current BiDictionary
  val inverse: BiMap[V, K] = i.getOrElse {
    val rev = m.map(_.swap)
    require((rev.size == m.size),
      s"Failed to create reversed map. Cannot have duplicated values.")
    new BiMap(rev, Some(this))
  }

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
    private val i: Option[BiMap[Int, String]] = None )
  extends BiMap[String, Int](m, i) {
}

/** BiDictionary is a specialized BiMap that has non-negative Ints as values for use as DRM keys.
  * The companion object provides modification methods specific to maintaining contiguous Int values
  * and unique String keys */
object BiDictionary {

  /**
   * Append new keys to an existing BiDictionary and return the result. The values will start
   * at m.size and increase to create a continuous non-zero value set from 0 to size - 1
   * @param keys new keys to append
   * @return a BiDictionary with added mappings
   */
  def append(keys: Set[String], biDi: BiDictionary): BiDictionary = {
    val hm = HashMap(keys.toSeq.view.zip (Stream from biDi.size): _*)
    new BiDictionary(biDi.m ++ hm)
  }

  /**
   * Append new keys to an existing BiDictionary and return the result. The values will start
   * at m.size and increase to create a continuous non-zero value set from 0 to size - 1
   * @param keys new keys to append
   * @return a BiDictionary with added mappings
   */
  def append(keys: List[String], biDi: BiDictionary): BiDictionary = {
    val hm = HashMap(keys.view.zip (Stream from biDi.size): _*)
    new BiDictionary(biDi.m ++ hm)
  }

  /**
   * Create a new BiDictionary with the keys supplied and values ranging from 0 to size -1
   * @param keys a set of String
   */
  def stringInt(keys: Set[String]): BiDictionary = {
    val hm = HashMap(keys.toSeq.view.zipWithIndex: _*)
    new BiDictionary(hm)
  }

  /**
   * Create a new empty BiDictionary, for Guava HashBiMap style instantiation.
   * todo: not really needed but requires using code to be changed--later.
   */
  def create(): BiDictionary = {
    val hm = HashMap[String, Int]()
    new BiDictionary(hm)
  }

  /**
   * todo: Create a BiDictionary from an RDD of keys, which will be made distinct first. Needs to be in
   *       Spark specific module. Or maybe a package helper function. This is so the RDD won't have to
   *       be collected into memory.
   */
}