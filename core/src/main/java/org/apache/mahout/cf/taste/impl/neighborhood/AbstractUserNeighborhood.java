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

package org.apache.mahout.cf.taste.impl.neighborhood;

import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.common.Refreshable;

import java.util.Collection;

/**
 * <p>Contains methods and resources useful to all classes in this package.</p>
 */
abstract class AbstractUserNeighborhood implements UserNeighborhood {

  private final UserCorrelation userCorrelation;
  private final DataModel dataModel;
  private final double samplingRate;
  private final RefreshHelper refreshHelper;

  AbstractUserNeighborhood(UserCorrelation userCorrelation,
                           DataModel dataModel,
                           double samplingRate) {
    if (userCorrelation == null || dataModel == null) {
      throw new IllegalArgumentException("userCorrelation or dataModel is null");
    }
    if (Double.isNaN(samplingRate) || samplingRate <= 0.0 || samplingRate > 1.0) {
      throw new IllegalArgumentException("samplingRate must be in (0,1]");
    }
    this.userCorrelation = userCorrelation;
    this.dataModel = dataModel;
    this.samplingRate = samplingRate;
    this.refreshHelper = new RefreshHelper(null);
    this.refreshHelper.addDependency(this.dataModel);
    this.refreshHelper.addDependency(this.userCorrelation);
  }

  final UserCorrelation getUserCorrelation() {
    return userCorrelation;
  }

  final DataModel getDataModel() {
    return dataModel;
  }

  final boolean sampleForUser() {
    return samplingRate >= 1.0 || Math.random() < samplingRate;
  }

  public final void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

}
