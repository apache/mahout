/*
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

package org.apache.mahout.cf.taste.hadoop;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.math.Varint;

/**
 * A {@link Writable} which encapsulates a list of {@link RecommendedItem}s. This is the mapper (and reducer)
 * output, and represents items recommended to a user. The first item is the one whose estimated preference is
 * highest.
 */
public final class RecommendedItemsWritable implements Writable {

  private List<RecommendedItem> recommended;
  
  public RecommendedItemsWritable() {
  // do nothing
  }
  
  public RecommendedItemsWritable(List<RecommendedItem> recommended) {
    this.recommended = recommended;
  }
  
  public List<RecommendedItem> getRecommendedItems() {
    return recommended;
  }

  public void set(List<RecommendedItem> recommended) {
    this.recommended = recommended;
  }
  
  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(recommended.size());
    for (RecommendedItem item : recommended) {
      Varint.writeSignedVarLong(item.getItemID(), out);
      out.writeFloat(item.getValue());
    }
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    int size = in.readInt();
    recommended = Lists.newArrayListWithCapacity(size);
    for (int i = 0; i < size; i++) {
      long itemID = Varint.readSignedVarLong(in);
      float value = in.readFloat();
      RecommendedItem recommendedItem = new GenericRecommendedItem(itemID, value);
      recommended.add(recommendedItem);
    }
  }
  
  @Override
  public String toString() {
    StringBuilder result = new StringBuilder(200);
    result.append('[');
    boolean first = true;
    for (RecommendedItem item : recommended) {
      if (first) {
        first = false;
      } else {
        result.append(',');
      }
      result.append(String.valueOf(item.getItemID()));
      result.append(':');
      result.append(String.valueOf(item.getValue()));
    }
    result.append(']');
    return result.toString();
  }
  
}
