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

package org.apache.mahout.math.drm

import scala.reflect.ClassTag
import org.apache.mahout.math._


/**
 * Additional experimental operations over CheckpointedDRM implementation. I will possibly move them up to
 * the DRMBase once they stabilize.
 *
 */
class CheckpointedOps[K: ClassTag](val drm: CheckpointedDrm[K]) {


  /** Column sums. At this point this runs on checkpoint and collects in-core vector. */
  def colSums(): Vector = drm.context.colSums(drm)

  /** Column clounts. Counts the non-zero values. At this point this runs on checkpoint and collects in-core vector. */
  def getNumNonZeroElements(): Vector = drm.context.getNumNonZeroElements(drm)

  /** Column Means */
  def colMeans(): Vector = drm.context.colMeans(drm)

  def norm():Double = drm.context.norm(drm)
}

