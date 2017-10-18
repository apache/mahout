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

package org.apache.mahout.classifier.email;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Iterator;

public class PrepEmailReducer extends Reducer<Text, VectorWritable, Text, VectorWritable> {

  private long maxItemsPerLabel = 10000;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    maxItemsPerLabel = Long.parseLong(context.getConfiguration().get(PrepEmailVectorsDriver.ITEMS_PER_CLASS));
  }

  @Override
  protected void reduce(Text key, Iterable<VectorWritable> values, Context context)
    throws IOException, InterruptedException {
    //TODO: support randomization?  Likely not needed due to the SplitInput utility which does random selection
    long i = 0;
    Iterator<VectorWritable> iterator = values.iterator();
    while (i < maxItemsPerLabel && iterator.hasNext()) {
      context.write(key, iterator.next());
      i++;
    }
  }
}
