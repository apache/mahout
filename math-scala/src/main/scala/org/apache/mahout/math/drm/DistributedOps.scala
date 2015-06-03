/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.math.drm

import org.apache.mahout.math.Vector

import scala.reflect.ClassTag

// Distributed operations given a DrmLike[K]
class DistributedOps[K: ClassTag] (protected[drm] val drm: DrmLike[K]){

  def accumulateBlocks[U: ClassTag](zeroValue: U, seqOp: (U, BlockifiedDrmTuple[K]) => U, combOp: (U, U) => U ): U =
    drm.context.operations.aggregateBlocks(drm)(zeroValue, seqOp, combOp)

  def accumulateRows[U: ClassTag](zeroValue: U, seqOp: (U, (K, Vector)) => U, combOp: (U, U) => U ): U =
    drm.context.operations.aggregateRows(drm)(zeroValue, seqOp, combOp)
}
