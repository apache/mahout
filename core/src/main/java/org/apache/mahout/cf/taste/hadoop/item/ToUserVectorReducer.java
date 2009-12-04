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

package org.apache.mahout.cf.taste.hadoop.item;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.ItemPrefWritable;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.matrix.Vector;

import java.io.IOException;
import java.util.Iterator;

public final class ToUserVectorReducer
    extends MapReduceBase
    implements Reducer<LongWritable, ItemPrefWritable, LongWritable, Vector> {

  @Override
  public void reduce(LongWritable userID,
                     Iterator<ItemPrefWritable> itemPrefs,
                     OutputCollector<LongWritable, Vector> output,
                     Reporter reporter) throws IOException {
    if (itemPrefs.hasNext()) {
      Vector userVector = new SparseVector();
      while (itemPrefs.hasNext()) {
        ItemPrefWritable itemPref = itemPrefs.next();
        userVector.set(String.valueOf(itemPref.getItemID()), itemPref.getPrefValue());
      }
      output.collect(userID, userVector);
    }
  }

}
