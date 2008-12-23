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
 * <p>{@link java.util.Comparator} that orders {@link org.apache.mahout.cf.taste.model.Preference}s by
 * {@link org.apache.mahout.cf.taste.model.User}.</p>
 */
public final class ByUserPreferenceComparator implements Comparator<Preference>, Serializable {

  private static final Comparator<Preference> instance = new ByUserPreferenceComparator();

  private ByUserPreferenceComparator() {
  }

  public static Comparator<Preference> getInstance() {
    return instance;
  }

  @Override
  public int compare(Preference o1, Preference o2) {
    return o1.getUser().compareTo(o2.getUser());
  }

  @Override
  public String toString() {
    return "ByUserPreferenceComparator";
  }

}
