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

import java.io.Serializable;

/**
 * <p>An {@link Item} which has no data other than an ID. This may be most useful for writing tests.</p>
 */
public class GenericItem<K extends Comparable<K>> implements Item, Serializable {

  private final K id;
  private final boolean recommendable;

  public GenericItem(K id) {
    this(id, true);
  }

  public GenericItem(K id, boolean recommendable) {
    if (id == null) {
      throw new IllegalArgumentException("id is null");
    }
    this.id = id;
    this.recommendable = recommendable;
  }

  public Object getID() {
    return id;
  }

  public boolean isRecommendable() {
    return recommendable;
  }

  @Override
  public int hashCode() {
    return id.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    return obj instanceof Item && ((Item) obj).getID().equals(id);
  }

  @Override
  public String toString() {
    return "Item[id:" + String.valueOf(id) + ']';
  }

  public int compareTo(Item item) {
    return id.compareTo((K) item.getID());
  }

}
