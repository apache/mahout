/**
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

package org.apache.mahout.clustering.meanshift;

public final class MeanShiftCanopyConfigKeys {

  private MeanShiftCanopyConfigKeys() {}

  // keys used by Driver, Mapper, Combiner & Reducer
  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.canopy.measure";
  public static final String KERNEL_PROFILE_KEY = "org.apache.mahout.clustering.canopy.kernelprofile";
  public static final String T1_KEY = "org.apache.mahout.clustering.canopy.t1";
  public static final String T2_KEY = "org.apache.mahout.clustering.canopy.t2";
  public static final String CONTROL_PATH_KEY = "org.apache.mahout.clustering.control.path";
  public static final String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.canopy.convergence";
  public static final String CLUSTER_POINTS_KEY = "org.apache.mahout.clustering.meanshift.clusterPointsKey";

}
