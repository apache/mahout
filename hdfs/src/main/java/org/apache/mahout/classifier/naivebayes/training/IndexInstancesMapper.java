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

package org.apache.mahout.classifier.naivebayes.training;

import java.io.IOException;
import java.util.regex.Pattern;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

public class IndexInstancesMapper extends Mapper<Text, VectorWritable, IntWritable, VectorWritable> {

  private static final Pattern SLASH = Pattern.compile("/");

  public enum Counter { SKIPPED_INSTANCES }

  private OpenObjectIntHashMap<String> labelIndex;

  @Override
  protected void setup(Context ctx) throws IOException, InterruptedException {
    super.setup(ctx);
    labelIndex = BayesUtils.readIndexFromCache(ctx.getConfiguration());
  }

  @Override
  protected void map(Text labelText, VectorWritable instance, Context ctx) throws IOException, InterruptedException {
    String label = SLASH.split(labelText.toString())[1];
    if (labelIndex.containsKey(label)) {
      ctx.write(new IntWritable(labelIndex.get(label)), instance);
    } else {
      ctx.getCounter(Counter.SKIPPED_INSTANCES).increment(1);
    }
  }
}
