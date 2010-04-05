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
import java.util.Iterator;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;

/**
 * Finally compute the cosine for each item-pair
 */
public final class CosineSimilarityReducer extends MapReduceBase
    implements Reducer<ItemPairWritable,FloatWritable,EntityEntityWritable,DoubleWritable> {

  @Override
  public void reduce(ItemPairWritable pair,
                     Iterator<FloatWritable> numeratorSummands,
                     OutputCollector<EntityEntityWritable,DoubleWritable> output,
                     Reporter reporter)
      throws IOException {

    double numerator = 0.0;
    while (numeratorSummands.hasNext()) {
      numerator += numeratorSummands.next().get();
    }
    double denominator = pair.getMultipliedLength();
    double cosine = numerator / denominator;
    output.collect(pair.getItemItemWritable(), new DoubleWritable(cosine));
  }

}
