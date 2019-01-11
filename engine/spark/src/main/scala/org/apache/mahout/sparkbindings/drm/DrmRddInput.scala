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
import org.apache.mahout.sparkbindings._

/** Encapsulates either DrmRdd[K] or BlockifiedDrmRdd[K] */
class DrmRddInput[K: ClassTag](private val input: Either[DrmRdd[K], BlockifiedDrmRdd[K]]) {

  private[sparkbindings] lazy val backingRdd = input.left.getOrElse(input.right.get)

  def isBlockified: Boolean = input.isRight

  def isRowWise: Boolean = input.isLeft

  def asRowWise(): DrmRdd[K] = input.left.getOrElse(deblockify(rdd = input.right.get))

  /** Use late binding for this. It may or may not be needed, depending on current config. */
  def asBlockified(ncol: ⇒ Int) = input.right.getOrElse(blockify(rdd = input.left.get, blockncol = ncol))

  def sparkContext: SparkContext = backingRdd.sparkContext

}
