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

package org.apache.mahout.math.drm.logical

import org.apache.mahout.math.drm.{DistributedContext, DrmLike}

/**
  * Any logical binary operator (such as A + B).
  * <P/>
  *
  * Any logical operator derived from this is also capabile of triggering optimizer checkpoint, hence,
  * it also inherits CheckpointAction.
  * <P/>
  *
  * @tparam A LHS key type
  * @tparam B RHS key type
  * @tparam K result key type
  */
abstract class AbstractBinaryOp[A, B, K]
  extends CheckpointAction[K] with DrmLike[K] {

  protected[drm] var A: DrmLike[A]

  protected[drm] var B: DrmLike[B]

  lazy val context: DistributedContext = A.context

  protected[mahout] def canHaveMissingRows: Boolean = false
}
