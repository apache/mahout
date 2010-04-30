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

package org.apache.mahout.clustering.fuzzykmeans;

public interface FuzzyKMeansConfigKeys {

  String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.kmeans.measure";

  String CLUSTER_PATH_KEY = "org.apache.mahout.clustering.kmeans.path";

  String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.kmeans.convergence";

  String M_KEY = "org.apache.mahout.clustering.fuzzykmeans.m";

  String EMIT_MOST_LIKELY_KEY = "org.apache.mahout.clustering.fuzzykmeans.emitMostLikely";

  String THRESHOLD_KEY = "org.apache.mahout.clustering.fuzzykmeans.threshold";

}
