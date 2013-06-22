package org.apache.mahout.vectorizer.pruner;
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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.apache.mahout.vectorizer.HighDFWordsPruner;

import java.io.IOException;
import java.util.Iterator;

public class WordsPrunerReducer extends
        Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

  private final OpenIntLongHashMap dictionary = new OpenIntLongHashMap();
  private long maxDf = Long.MAX_VALUE;
  private long minDf = -1;

  @Override
  protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context context)
    throws IOException, InterruptedException {
    Iterator<VectorWritable> it = values.iterator();
    if (!it.hasNext()) {
      return;
    }
    Vector value = it.next().get();
    Vector vector = value.clone();
    if (maxDf != Long.MAX_VALUE || minDf > -1) {
      for (Vector.Element e : value.nonZeroes()) {
        if (!dictionary.containsKey(e.index())) {
          vector.setQuick(e.index(), 0.0);
          continue;
        }
        long df = dictionary.get(e.index());
        if (df > maxDf || df < minDf) {
          vector.setQuick(e.index(), 0.0);
        }
      }
    }

    VectorWritable vectorWritable = new VectorWritable(vector);
    context.write(key, vectorWritable);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    //Path[] localFiles = HadoopUtil.getCachedFiles(conf);

    maxDf = conf.getLong(HighDFWordsPruner.MAX_DF, Long.MAX_VALUE);
    minDf = conf.getLong(HighDFWordsPruner.MIN_DF, -1);

    Path dictionaryFile = HadoopUtil.getSingleCachedFile(conf);

    // key is feature, value is the document frequency
    for (Pair<IntWritable, LongWritable> record
            : new SequenceFileIterable<IntWritable, LongWritable>(dictionaryFile, true, conf)) {
      dictionary.put(record.getFirst().get(), record.getSecond().get());
    }
  }
}
