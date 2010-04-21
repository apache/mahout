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

package org.apache.mahout.utils.vectors.common;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Merges partial vectors in to a full sparse vector
 */
public class PartialVectorMergeReducer extends MapReduceBase implements
    Reducer<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {

  private double normPower;
  private int dimension;
  private boolean sequentialAccess;
  
  @Override
  public void reduce(WritableComparable<?> key,
                     Iterator<VectorWritable> values,
                     OutputCollector<WritableComparable<?>,VectorWritable> output,
                     Reporter reporter) throws IOException {
    
    Vector vector = new RandomAccessSparseVector(dimension, 10);
    while (values.hasNext()) {
      VectorWritable value = values.next();
      value.get().addTo(vector);
    }
    if (normPower != PartialVectorMerger.NO_NORMALIZING) {
      vector = vector.normalize(normPower);
    }
    if (sequentialAccess) {
      vector = new SequentialAccessSparseVector(vector);
    }
    VectorWritable vectorWritable = new VectorWritable(vector);
    output.collect(key, vectorWritable);
  }
  
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    normPower = job.getFloat(PartialVectorMerger.NORMALIZATION_POWER, PartialVectorMerger.NO_NORMALIZING);
    dimension = job.getInt(PartialVectorMerger.DIMENSION, Integer.MAX_VALUE);
    sequentialAccess = job.getBoolean(PartialVectorMerger.SEQUENTIAL_ACCESS, false);
  }
}
