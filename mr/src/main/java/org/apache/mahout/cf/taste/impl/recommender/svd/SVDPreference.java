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

package org.apache.mahout.cf.taste.impl.recommender.svd;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;

final class SVDPreference extends GenericPreference {

  private double cache;

  SVDPreference(long userID, long itemID, float value, double cache) {
    super(userID, itemID, value);
    setCache(cache);
  }

  public double getCache() {
    return cache;
  }

  public void setCache(double value) {
    Preconditions.checkArgument(!Double.isNaN(value), "NaN cache value");
    this.cache = value;
  }

}
