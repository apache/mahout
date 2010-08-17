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

package org.apache.mahout.utils.vectors.tfidf;

import java.io.IOException;
import java.net.URI;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.apache.mahout.utils.vectors.TFIDF;
import org.apache.mahout.utils.vectors.common.PartialVectorMerger;

/**
 * Converts a document in to a sparse vector
 */
public class TFIDFPartialVectorReducer extends
    Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

  private final OpenIntLongHashMap dictionary = new OpenIntLongHashMap();

  private final TFIDF tfidf = new TFIDF();

  private int minDf = 1;

  private int maxDfPercent = 99;

  private long vectorCount = 1;

  private long featureCount;

  private boolean sequentialAccess;

  @Override
  protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context context)
      throws IOException, InterruptedException {
    Iterator<VectorWritable> it = values.iterator();
    if (!it.hasNext()) {
      return;
    }
    Vector value = it.next().get();
    Iterator<Vector.Element> it1 = value.iterateNonZero();
    Vector vector = new RandomAccessSparseVector((int) featureCount, value.getNumNondefaultElements());
    while (it1.hasNext()) {
      Vector.Element e = it1.next();
      if (!dictionary.containsKey(e.index())) {
        continue;
      }
      long df = dictionary.get(e.index());
      if (df / vectorCount > maxDfPercent) {
        continue;
      }
      if (df < minDf) {
        df = minDf;
      }
      vector.setQuick(e.index(), tfidf.calculate((int) e.get(), (int) df, (int) featureCount, (int) vectorCount));
    }
    if (sequentialAccess) {
      vector = new SequentialAccessSparseVector(vector);
    }
    VectorWritable vectorWritable = new VectorWritable(new NamedVector(vector, key.toString()));
    context.write(key, vectorWritable);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    try {
      Configuration conf = context.getConfiguration();
      URI[] localFiles = DistributedCache.getCacheFiles(conf);
      if (localFiles == null || localFiles.length < 1) {
        throw new IllegalArgumentException("missing paths from the DistributedCache");
      }

      vectorCount = conf.getLong(TFIDFConverter.VECTOR_COUNT, 1);
      featureCount = conf.getLong(TFIDFConverter.FEATURE_COUNT, 1);
      minDf = conf.getInt(TFIDFConverter.MIN_DF, 1);
      maxDfPercent = conf.getInt(TFIDFConverter.MAX_DF_PERCENTAGE, 99);
      sequentialAccess = conf.getBoolean(PartialVectorMerger.SEQUENTIAL_ACCESS, false);

      Path dictionaryFile = new Path(localFiles[0].getPath());
      FileSystem fs = dictionaryFile.getFileSystem(conf);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, dictionaryFile, conf);
      IntWritable key = new IntWritable();
      LongWritable value = new LongWritable();

      // key is feature, value is the document frequency
      while (reader.next(key, value)) {
        dictionary.put(key.get(), value.get());
      }
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

}
