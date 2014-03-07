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

package org.apache.mahout.sparkbindings.drm

import scala.reflect.ClassTag
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel

/** Encapsulates either DrmRdd[K] or BlockifiedDrmRdd[K] */
class DrmRddInput[K: ClassTag](
    private val rowWiseSrc: Option[( /*ncol*/ Int, /*rdd*/ DrmRdd[K])] = None,
    private val blockifiedSrc: Option[BlockifiedDrmRdd[K]] = None
    ) {

  assert(rowWiseSrc.isDefined || blockifiedSrc.isDefined, "Undefined input")

  private lazy val backingRdd = rowWiseSrc.map(_._2).getOrElse(blockifiedSrc.get)

  def toDrmRdd(): DrmRdd[K] = rowWiseSrc.map(_._2).getOrElse(deblockify(rdd = blockifiedSrc.get))

  def toBlockifiedDrmRdd() = blockifiedSrc.getOrElse(blockify(rdd = rowWiseSrc.get._2, blockncol = rowWiseSrc.get._1))

  def sparkContext: SparkContext = backingRdd.sparkContext

  def persist(sl: StorageLevel) = backingRdd.persist(newLevel = sl)

  def unpersist(blocking: Boolean = true) = backingRdd.unpersist(blocking)
}
