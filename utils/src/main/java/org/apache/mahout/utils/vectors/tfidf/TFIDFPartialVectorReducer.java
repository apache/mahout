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

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.apache.mahout.utils.vectors.TFIDF;

/**
 * Converts a document in to a sparse vector
 */
public class TFIDFPartialVectorReducer extends MapReduceBase
    implements
    Reducer<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {
  
  private final OpenIntLongHashMap dictionary = new OpenIntLongHashMap();
  private final VectorWritable vectorWritable = new VectorWritable();
  private final TFIDF tfidf = new TFIDF();
  private int minDf = 1;
  private int maxDfPercent = 99;
  private long vectorCount = 1;
  private long featureCount = 0;
  
  @Override
  public void reduce(WritableComparable<?> key,
                     Iterator<VectorWritable> values,
                     OutputCollector<WritableComparable<?>,VectorWritable> output,
                     Reporter reporter) throws IOException {
    if (!values.hasNext()) return;
    Vector value = values.next().get();
    Iterator<Element> it = value.iterateNonZero();
    Vector vector = new RandomAccessSparseVector(key
        .toString(), Integer.MAX_VALUE, value.getNumNondefaultElements());
    while (it.hasNext()) {
      Element e = it.next();
      if (!dictionary.containsKey(e.index())) continue;
      long df = dictionary.get(e.index());
      if (df / vectorCount > maxDfPercent) continue;
      if (df < minDf) df = minDf;
      vector.setQuick(e.index(), tfidf.calculate((int) e.get(), (int) df,
        (int) featureCount, (int) vectorCount));
    }
    
    vectorWritable.set(vector);
    output.collect(key, vectorWritable);
  }
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    try {
      
      URI[] localFiles = DistributedCache.getCacheFiles(job);
      if (localFiles == null || localFiles.length < 1) {
        throw new IllegalArgumentException(
            "missing paths from the DistributedCache");
      }
      
      vectorCount = job.getLong(TFIDFConverter.VECTOR_COUNT, 1);
      featureCount = job.getLong(TFIDFConverter.FEATURE_COUNT, 1);
      minDf = job.getInt(TFIDFConverter.MIN_DF, 1);
      maxDfPercent = job.getInt(TFIDFConverter.MAX_DF_PERCENTAGE, 99);
      
      Path dictionaryFile = new Path(localFiles[0].getPath());
      FileSystem fs = dictionaryFile.getFileSystem(job);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, dictionaryFile,
          job);
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
