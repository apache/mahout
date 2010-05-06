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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.similarity.CoRating;

/**
 * map out each pair of items that appears in the same user-vector together with the multiplied vector lengths
 * of the associated item vectors
 */
public final class CopreferredItemsMapper extends MapReduceBase
    implements Mapper<LongWritable,ItemPrefWithItemVectorWeightArrayWritable,ItemPairWritable,CoRating> {

  @Override
  public void map(LongWritable user,
                  ItemPrefWithItemVectorWeightArrayWritable itemPrefsArray,
                  OutputCollector<ItemPairWritable, CoRating> output,
                  Reporter reporter)
      throws IOException {

    ItemPrefWithItemVectorWeightWritable[] itemPrefs = itemPrefsArray.getItemPrefs();

    for (int n = 0; n < itemPrefs.length; n++) {
      ItemPrefWithItemVectorWeightWritable itemN = itemPrefs[n];
      long itemNID = itemN.getItemID();
      double itemNWeight = itemN.getWeight();
      float itemNValue = itemN.getPrefValue();
      for (int m = n + 1; m < itemPrefs.length; m++) {
        ItemPrefWithItemVectorWeightWritable itemM = itemPrefs[m];
        long itemAID = Math.min(itemNID, itemM.getItemID());
        long itemBID = Math.max(itemNID, itemM.getItemID());
        ItemPairWritable pair = new ItemPairWritable(itemAID, itemBID, itemNWeight, itemM.getWeight());
        output.collect(pair, new CoRating(itemNValue, itemM.getPrefValue()));
      }
    }

  }


}
