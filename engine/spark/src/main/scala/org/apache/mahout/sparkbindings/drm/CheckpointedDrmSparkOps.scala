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

import org.apache.mahout.math.drm.CheckpointedDrm
import org.apache.mahout.sparkbindings.DrmRdd

/** Additional Spark-specific operations. Requires underlying DRM to be running on Spark backend. */
class CheckpointedDrmSparkOps[K](drm: CheckpointedDrm[K]) {

  assert(drm.isInstanceOf[CheckpointedDrmSpark[K]], "must be a Spark-backed matrix")

  private[sparkbindings] val sparkDrm = drm.asInstanceOf[CheckpointedDrmSpark[K]]

  /** Spark matrix customization exposure */
  def rdd = sparkDrm.rddInput.asRowWise()

}
