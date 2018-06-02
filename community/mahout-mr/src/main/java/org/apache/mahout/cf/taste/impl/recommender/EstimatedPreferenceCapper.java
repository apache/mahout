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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.model.DataModel;

/**
 * Simple class which encapsulates restricting a preference value
 * to a predefined range. The simple logic is wrapped up here for
 * performance reasons.
 */
public final class EstimatedPreferenceCapper {

  private final float min;
  private final float max;

  public EstimatedPreferenceCapper(DataModel model) {
    min = model.getMinPreference();
    max = model.getMaxPreference();
  }

  public float capEstimate(float estimate) {
    if (estimate > max) {
      estimate = max;
    } else if (estimate < min) {
      estimate = min;
    }
    return estimate;
  }

}
