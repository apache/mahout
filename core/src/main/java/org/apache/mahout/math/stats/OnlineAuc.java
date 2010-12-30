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

package org.apache.mahout.math.stats;

import org.apache.hadoop.io.Writable;

/**
 * Describes the generic outline of how to compute AUC.  Currently there are two
 * implementations of this, one for computing a global estimate of AUC and the other
 * for computing average grouped AUC.  Grouped AUC is useful when misusing a classifier
 * as a recommendation system.
 */
public interface OnlineAuc extends Writable {
  double addSample(int category, String groupKey, double score);

  double addSample(int category, double score);

  double auc();

  void setPolicy(GlobalOnlineAuc.ReplacementPolicy policy);

  void setWindowSize(int windowSize);
}
