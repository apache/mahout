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

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * <h1>Input</h1>
 * 
 * <p>
 * Takes user IDs as {@link LongWritable} mapped to all associated item IDs and preference values, as
 * {@link EntityPrefWritable}s.
 * </p>
 * 
 * <h1>Output</h1>
 * 
 * <p>
 * The same user ID mapped to a {@link RandomAccessSparseVector} representation of the same item IDs and
 * preference values. Item IDs are used as vector indexes; they are hashed into ints to work as indexes with
 * {@link ItemIDIndexMapper#idToIndex(long)}. The mapping is remembered for later with a combination of
 * {@link ItemIDIndexMapper} and {@link ItemIDIndexReducer}.
 * </p>
 */
public final class ToUserVectorReducer extends MapReduceBase implements
    Reducer<LongWritable,LongWritable,LongWritable, VectorWritable> {
  
  @Override
  public void reduce(LongWritable userID,
                     Iterator<LongWritable> itemPrefs,
                     OutputCollector<LongWritable,VectorWritable> output,
                     Reporter reporter) throws IOException {
    if (!itemPrefs.hasNext()) {
      return;
    }
    Vector userVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    while (itemPrefs.hasNext()) {
      LongWritable itemPref = itemPrefs.next();
      int index = ItemIDIndexMapper.idToIndex(itemPref.get());
      float value;
      if (itemPref instanceof EntityPrefWritable) {
        value = ((EntityPrefWritable) itemPref).getPrefValue();
      } else {
        value = 1.0f;
      }
      userVector.set(index, value);
    }

    output.collect(userID, new VectorWritable(userVector));
  }
  
}
