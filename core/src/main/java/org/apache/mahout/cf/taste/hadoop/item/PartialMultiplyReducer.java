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

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.LongFloatProcedure;
import org.apache.mahout.math.map.OpenLongFloatHashMap;

public final class PartialMultiplyReducer extends MapReduceBase implements
    Reducer<IntWritable,VectorOrPrefWritable,LongWritable,VectorWritable> {

  @Override
  public void reduce(IntWritable key,
                     Iterator<VectorOrPrefWritable> values,
                     final OutputCollector<LongWritable,VectorWritable> output,
                     Reporter reporter) throws IOException {

    OpenLongFloatHashMap savedValues = new OpenLongFloatHashMap();
    Vector cooccurrenceColumn = null;
    final int itemIndex = key.get();
    final LongWritable userIDWritable = new LongWritable();
    final VectorWritable vectorWritable = new VectorWritable();

    while (values.hasNext()) {

      VectorOrPrefWritable value = values.next();
      if (value.getVector() == null) {

        // Then this is a user-pref value
        long userID = value.getUserID();
        float preferenceValue = value.getValue();
        
        if (cooccurrenceColumn == null) {
          // Haven't seen the co-occurrencce column yet; save it
          savedValues.put(userID, preferenceValue);
        } else {
          // Have seen it
          Vector partialProduct = cooccurrenceColumn.times(preferenceValue);
          // This makes sure this item isn't recommended for this user:
          partialProduct.set(itemIndex, Double.NEGATIVE_INFINITY);
          userIDWritable.set(userID);
          vectorWritable.set(partialProduct);
          output.collect(userIDWritable, vectorWritable);
        }

      } else {

        // Then this is the column vector
        cooccurrenceColumn = value.getVector();

        final Vector theColumn = cooccurrenceColumn;
        savedValues.forEachPair(new LongFloatProcedure() {
          @Override
          public boolean apply(long userID, float value) {
            Vector partialProduct = theColumn.times(value);
            // This makes sure this item isn't recommended for this user:
            partialProduct.set(itemIndex, Double.NEGATIVE_INFINITY);
            userIDWritable.set(userID);
            vectorWritable.set(partialProduct);
            try {
              output.collect(userIDWritable, vectorWritable);
            } catch (IOException ioe) {
              throw new IllegalStateException(ioe);
            }
            return true;
          }
        });
        savedValues.clear();
      }
    }

  }

}