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
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Iterator;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * <h1>Input</h1>
 *
 * <p>Takes user IDs as {@link LongWritable} mapped to all associated item IDs
 * and preference values, as {@link ItemPrefWritable}s.</p>
 *
 * <h1>Output</h1>
 *
 * <p>The same user ID mapped to a {@link RandomAccessSparseVector} representation of the
 * same item IDs and preference values. Item IDs are used as vector indexes;
 * they are hashed into ints to work as indexes with
 * {@link ItemIDIndexMapper#idToIndex(long)}. The mapping is remembered for
 * later with a combination of {@link ItemIDIndexMapper} and {@link ItemIDIndexReducer}.</p>
 *
 * <p>The number of non-default elements actually retained in the user vector is capped
 * at {@link #MAX_PREFS_CONSIDERED}.</p>
 *
 */
public final class ToUserVectorReducer
    extends MapReduceBase
    implements Reducer<LongWritable, ItemPrefWritable, LongWritable, VectorWritable> {

  public static final int MAX_PREFS_CONSIDERED = 20;

  private final VectorWritable vectorWritable = new VectorWritable();

  @Override
  public void reduce(LongWritable userID,
                     Iterator<ItemPrefWritable> itemPrefs,
                     OutputCollector<LongWritable, VectorWritable> output,
                     Reporter reporter) throws IOException {
    if (itemPrefs.hasNext()) {
      RandomAccessSparseVector userVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
      while (itemPrefs.hasNext()) {
        ItemPrefWritable itemPref = itemPrefs.next();
        int index = ItemIDIndexMapper.idToIndex(itemPref.getItemID());
        userVector.set(index, itemPref.getPrefValue());
      }

      if (userVector.getNumNondefaultElements() > MAX_PREFS_CONSIDERED) {
        double cutoff = findTopNPrefsCutoff(MAX_PREFS_CONSIDERED, userVector);
        RandomAccessSparseVector filteredVector = new RandomAccessSparseVector(Integer.MAX_VALUE, MAX_PREFS_CONSIDERED);
        Iterator<Vector.Element> it = userVector.iterateNonZero();
        while (it.hasNext()) {
          Vector.Element element = it.next();
          if (element.get() >= cutoff) {
            filteredVector.set(element.index(), element.get());
          }
        }
        userVector = filteredVector;
      }

      vectorWritable.set(userVector);
      output.collect(userID, vectorWritable);
    }
  }

  private static double findTopNPrefsCutoff(int n, Vector userVector) {
    Queue<Double> topPrefValues = new PriorityQueue<Double>(n + 1);
    Iterator<Vector.Element> it = userVector.iterateNonZero();
    while (it.hasNext()) {
      double prefValue = it.next().get();
      if (topPrefValues.size() < n) {
        topPrefValues.add(prefValue);
      } else if (prefValue > topPrefValues.peek()) {
        topPrefValues.add(prefValue);
        topPrefValues.poll();
      }
    }
    return topPrefValues.peek();
  }

}
