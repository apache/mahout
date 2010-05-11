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
import java.util.PriorityQueue;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.LongFloatProcedure;
import org.apache.mahout.math.function.LongProcedure;
import org.apache.mahout.math.map.OpenLongFloatHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class PartialMultiplyReducer extends MapReduceBase implements
    Reducer<IntWritable,VectorOrPrefWritable,VLongWritable,VectorWritable> {

  private static final Logger log = LoggerFactory.getLogger(PartialMultiplyReducer.class);

  private static final int MAX_PRODUCTS_PER_ITEM = 100;

  private enum Counters {
    PRODUCTS_OUTPUT,
    PRODUCTS_SKIPPED,
  }

  @Override
  public void reduce(IntWritable key,
                     Iterator<VectorOrPrefWritable> values,
                     final OutputCollector<VLongWritable,VectorWritable> output,
                     final Reporter reporter) {

    int itemIndex = key.get();
    OpenLongFloatHashMap savedValues = new OpenLongFloatHashMap();

    Vector cooccurrenceColumn = null;
    while (values.hasNext()) {
      VectorOrPrefWritable value = values.next();
      if (value.getVector() == null) {
        // Then this is a user-pref value
        savedValues.put(value.getUserID(), value.getValue());
      } else {
        // Then this is the column vector
        if (cooccurrenceColumn != null) {
          throw new IllegalStateException("Found two co-occurrence columns for item index " + itemIndex);
        }
        cooccurrenceColumn = value.getVector();
      }
    }

    final VLongWritable userIDWritable = new VLongWritable();

    // These single-element vectors ensure that each user will not be recommended
    // this item
    Vector excludeVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
    excludeVector.set(itemIndex, Double.NaN);
    final VectorWritable excludeWritable = new VectorWritable(excludeVector);
    excludeWritable.setWritesLaxPrecision(true);
    savedValues.forEachKey(new LongProcedure() {
      @Override
      public boolean apply(long userID) {
        userIDWritable.set(userID);
        try {
          output.collect(userIDWritable, excludeWritable);
        } catch (IOException ioe) {
          throw new IllegalStateException(ioe);
        }
        return true;
      }
    });

    if (cooccurrenceColumn == null) {
      log.info("Column vector missing for {}; continuing", itemIndex);
      return;
    }    

    final float smallestLargeValue = findSmallestLargeValue(savedValues);

    final VectorWritable vectorWritable = new VectorWritable();
    vectorWritable.setWritesLaxPrecision(true);

    final Vector theColumn = cooccurrenceColumn;
    savedValues.forEachPair(new LongFloatProcedure() {
      @Override
      public boolean apply(long userID, float value) {
        if (Math.abs(value) < smallestLargeValue) {
          reporter.incrCounter(Counters.PRODUCTS_SKIPPED, 1L);
        } else {
          try {
            Vector partialProduct = value == 1.0f ? theColumn : theColumn.times(value);
            userIDWritable.set(userID);
            vectorWritable.set(partialProduct);
            output.collect(userIDWritable, vectorWritable);
          } catch (IOException ioe) {
            throw new IllegalStateException(ioe);
          }
          reporter.incrCounter(Counters.PRODUCTS_OUTPUT, 1L);
        }
        return true;
      }
    });

  }

  private static float findSmallestLargeValue(OpenLongFloatHashMap savedValues) {
    final PriorityQueue<Float> topPrefValues = new PriorityQueue<Float>(MAX_PRODUCTS_PER_ITEM + 1);
    savedValues.forEachPair(new LongFloatProcedure() {
      @Override
      public boolean apply(long userID, float value) {
        if (topPrefValues.size() < MAX_PRODUCTS_PER_ITEM) {
          topPrefValues.add(value);
        } else {
          float absValue = Math.abs(value);
          if (absValue > topPrefValues.peek()) {
            topPrefValues.add(absValue);
            topPrefValues.poll();
          }
        }
        return true;
      }
    });
    return topPrefValues.peek();
  }

}