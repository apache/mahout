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

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

/**
 * map out each pair of items that appears in the same user-vector together with the multiplied vector lengths
 * of the associated item vectors
 */
public final class CopreferredItemsMapper extends MapReduceBase
    implements Mapper<LongWritable,ItemPrefWithLengthArrayWritable,ItemPairWritable,FloatWritable> {

  @Override
  public void map(LongWritable user,
                  ItemPrefWithLengthArrayWritable itemPrefsArray,
                  OutputCollector<ItemPairWritable,FloatWritable> output,
                  Reporter reporter)
      throws IOException {

    ItemPrefWithLengthWritable[] itemPrefs = itemPrefsArray.getItemPrefs();

    for (int n = 0; n < itemPrefs.length; n++) {
      ItemPrefWithLengthWritable itemN = itemPrefs[n];
      long itemNID = itemN.getItemID();
      double itemNLength = itemN.getLength();
      float itemNValue = itemN.getPrefValue();
      for (int m = n + 1; m < itemPrefs.length; m++) {
        ItemPrefWithLengthWritable itemM = itemPrefs[m];
        long itemAID = Math.min(itemNID, itemM.getItemID());
        long itemBID = Math.max(itemNID, itemM.getItemID());
        ItemPairWritable pair = new ItemPairWritable(itemAID, itemBID, itemNLength * itemM.getLength());
        output.collect(pair, new FloatWritable(itemNValue * itemM.getPrefValue()));
      }
    }

  }


}
