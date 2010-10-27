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

package org.apache.mahout.cf.taste.impl.model;

import org.apache.mahout.cf.taste.model.DataModel;

/**
 * Contains some features common to all implementations.
 */
public abstract class AbstractDataModel implements DataModel {

  private float maxPreference;
  private float minPreference;

  protected AbstractDataModel() {
    maxPreference = Float.NaN;
    minPreference = Float.NaN;
  }

  @Override
  public float getMaxPreference() {
    return maxPreference;
  }

  protected void setMaxPreference(float maxPreference) {
    this.maxPreference = maxPreference;
  }

  @Override
  public float getMinPreference() {
    return minPreference;
  }

  protected void setMinPreference(float minPreference) {
    this.minPreference = minPreference;
  }

}
