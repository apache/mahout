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

package org.apache.mahout.cf.taste.model;

import java.io.Serializable;

/**
 * An alternate representation of an array of {@link Preference}. Implementations, in theory, can produce a more
 * memory-efficient representation. This is not used yet.
 */
public interface PreferenceArray extends Cloneable, Serializable, Iterable<Preference> {

  int length();
  
  Preference get(int i);

  void set(int i, Preference pref);

  Comparable<?> getUserID(int i);

  void setUserID(int i, Comparable<?> userID);

  Comparable<?> getItemID(int i);

  void setItemID(int i, Comparable<?> itemID);

  float getValue(int i);

  void setValue(int i, float value);

  PreferenceArray clone();

  void sortByUser();

  void sortByItem();

  void sortByValue();

  void sortByValueReversed();

}