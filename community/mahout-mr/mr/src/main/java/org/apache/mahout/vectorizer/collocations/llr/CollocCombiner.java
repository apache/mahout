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

package org.apache.mahout.vectorizer.collocations.llr;

import java.io.IOException;

import org.apache.hadoop.mapreduce.Reducer;

/** Combiner for pass1 of the CollocationDriver. Combines frequencies for values for the same key */
public class CollocCombiner extends Reducer<GramKey, Gram, GramKey, Gram> {

  @Override
  protected void reduce(GramKey key, Iterable<Gram> values, Context context) throws IOException, InterruptedException {

    int freq = 0;
    Gram value = null;

    // accumulate frequencies from values, preserve the last value
    // to write to the context.
    for (Gram value1 : values) {
      value = value1;
      freq += value.getFrequency();
    }

    if (value != null) {
      value.setFrequency(freq);
      context.write(key, value);
    }
  }

}
