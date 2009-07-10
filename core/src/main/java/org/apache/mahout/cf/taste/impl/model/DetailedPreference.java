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

import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;

import java.text.DateFormat;
import java.util.Date;

/**
 * <p>An expanded version of {@link GenericPreference} which adds more fields;  for now, this only includes an
 * additional timestamp field. This is provided as a convenience to implementations and {@link
 * org.apache.mahout.cf.taste.model.DataModel}s which wish to record and use this information in computations. This
 * information is not added to {@link org.apache.mahout.cf.taste.impl.model.GenericPreference} to avoid expanding memory
 * requirements of the algorithms supplied with Taste, since memory is a limiting factor.</p>
 */
public class DetailedPreference extends GenericPreference {

  private final long timestamp;

  public DetailedPreference(User user, Item item, double value, long timestamp) {
    super(user, item, value);
    if (timestamp < 0L) {
      throw new IllegalArgumentException("timestamp is negative");
    }
    this.timestamp = timestamp;
  }

  public long getTimestamp() {
    return timestamp;
  }

  @Override
  public String toString() {
    return "GenericPreference[user: " + getUser() + ", item:" + getItem() + ", value:" + getValue() +
        ", timestamp: " + DateFormat.getDateTimeInstance().format(new Date(timestamp)) + ']';
  }

}
