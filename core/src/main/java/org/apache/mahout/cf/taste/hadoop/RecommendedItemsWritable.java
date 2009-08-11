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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.EOFException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link Writable} which encapsulates a list of {@link RecommendedItem}s. This is the mapper (and reducer) output,
 * and represents items recommended to a user. The first item is the one whose estimated preference is highest.
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

  @Override
  public void write(DataOutput out) throws IOException {
    for (RecommendedItem item : recommended) {
      out.writeLong(item.getItemID());
      out.writeFloat(item.getValue());
    }

  }

  @Override
  public void readFields(DataInput in) throws IOException {
    recommended = new ArrayList<RecommendedItem>();
    try {
      do {
        long itemID = in.readLong();
        float value = in.readFloat();
        RecommendedItem recommendedItem = new GenericRecommendedItem(itemID, value);
        recommended.add(recommendedItem);
      } while (true);
    } catch (EOFException eofe) {
      // continue; done
    }
  }

  public static RecommendedItemsWritable read(DataInput in) throws IOException {
    RecommendedItemsWritable writable = new RecommendedItemsWritable();
    writable.readFields(in);
    return writable;
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    result.append('[');
    boolean first = true;
    for (RecommendedItem item : recommended) {
      if (first) {
        first = false;
      } else {
        result.append(',');
      }
      result.append(item.getItemID());
      result.append(':');
      result.append(item.getValue());
    }
    result.append(']');
    return result.toString();
  }

}
