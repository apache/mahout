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

import org.apache.mahout.cf.taste.correlation.ItemCorrelation;
import org.apache.mahout.cf.taste.correlation.PreferenceInferrer;
import org.apache.mahout.cf.taste.correlation.UserCorrelation;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.common.Refreshable;

import java.util.Collection;

final class DummyCorrelation implements UserCorrelation, ItemCorrelation {

  public double userCorrelation(User user1, User user2) {
    return 1.0 / Math.abs(user1.getPreferencesAsArray()[0].getValue() -
                          user2.getPreferencesAsArray()[0].getValue());
  }

  public double itemCorrelation(Item item1, Item item2) {
    // Make up something wacky
    return (double) (item1.hashCode() - item2.hashCode());
  }

  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    throw new UnsupportedOperationException();
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
