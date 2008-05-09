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

import org.apache.mahout.cf.taste.model.Preference;

import java.io.Serializable;
import java.util.Comparator;

/**
 * <p>{@link Comparator} that orders {@link org.apache.mahout.cf.taste.model.Preference}s from least preferred
 * to most preferred -- that is, in order of ascending value.</p>
 */
public final class ByValuePreferenceComparator implements Comparator<Preference>, Serializable {

  private static final Comparator<Preference> instance = new ByValuePreferenceComparator();

  private ByValuePreferenceComparator() {
  }

  public static Comparator<Preference> getInstance() {
    return instance;
  }

  public int compare(Preference o1, Preference o2) {
    double value1 = o1.getValue();
    double value2 = o2.getValue();
    if (value1 < value2) {
      return -1;
    } else if (value1 > value2) {
      return 1;
    } else {
      return 0;
    }
  }

  @Override
  public String toString() {
    return "ByValuePreferenceComparator";
  }

}
