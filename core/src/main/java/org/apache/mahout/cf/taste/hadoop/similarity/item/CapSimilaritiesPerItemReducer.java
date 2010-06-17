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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;

/**
 * this reducer sees all similar items for an item in descending similarity value order and writes maximally as much
 * as specified in the "maxSimilaritiesPerItem" option of {@link ItemSimilarityJob}
 */
public class CapSimilaritiesPerItemReducer
    extends Reducer<CapSimilaritiesPerItemKeyWritable,SimilarItemWritable,EntityEntityWritable,DoubleWritable> {

  private int maxSimilaritiesPerItem;

  @Override
  protected void setup(Context ctx) throws IOException, InterruptedException {
    super.setup(ctx);
    maxSimilaritiesPerItem = ctx.getConfiguration().getInt(ItemSimilarityJob.MAX_SIMILARITIES_PER_ITEM, -1);
    if (maxSimilaritiesPerItem < 1) {
      throw new IllegalStateException("Maximum similar items per item was not set correctly");
    }
  }

  @Override
  protected void reduce(CapSimilaritiesPerItemKeyWritable capKey, Iterable<SimilarItemWritable> similarItems,
      Context ctx) throws IOException, InterruptedException {
    long itemAID = capKey.getItemID();

    /* we see the similar items in descending value order because of secondary sort */
    int n=0;
    for (SimilarItemWritable similarItem : similarItems) {
      long itemBID = similarItem.getItemID();
      EntityEntityWritable itemPair = toItemPair(itemAID, itemBID);
      ctx.write(itemPair, new DoubleWritable(similarItem.getValue()));

      if (++n == maxSimilaritiesPerItem) {
        break;
      }
    }
  }

  protected EntityEntityWritable toItemPair(long itemAID, long itemBID) {
    /* smaller ID first */
    if (itemAID < itemBID) {
      return new EntityEntityWritable(itemAID, itemBID);
    } else {
      return new EntityEntityWritable(itemBID, itemAID);
    }
  }
}
