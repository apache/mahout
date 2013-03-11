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

package org.apache.mahout.vectorizer.common;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

/**
 * Merges partial vectors in to a full sparse vector
 */
public class PartialVectorMergeReducer extends
    Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

  private double normPower;

  private int dimension;

  private boolean sequentialAccess;

  private boolean namedVector;
  
  private boolean logNormalize;
  
  @Override
  protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context context) throws IOException,
      InterruptedException {

    Vector vector = new RandomAccessSparseVector(dimension, 10);
    for (VectorWritable value : values) {
      vector.assign(value.get(), Functions.PLUS);
    }
    if (normPower != PartialVectorMerger.NO_NORMALIZING) {
      if (logNormalize) {
        vector = vector.logNormalize(normPower);
      } else {
        vector = vector.normalize(normPower);
      }
    }
    if (sequentialAccess) {
      vector = new SequentialAccessSparseVector(vector);
    }
    
    if (namedVector) {
      vector = new NamedVector(vector, key.toString());
    }

    // drop empty vectors.
    if (vector.getNumNondefaultElements() > 0) {
      VectorWritable vectorWritable = new VectorWritable(vector);
      context.write(key, vectorWritable);
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    normPower = conf.getFloat(PartialVectorMerger.NORMALIZATION_POWER, PartialVectorMerger.NO_NORMALIZING);
    dimension = conf.getInt(PartialVectorMerger.DIMENSION, Integer.MAX_VALUE);
    sequentialAccess = conf.getBoolean(PartialVectorMerger.SEQUENTIAL_ACCESS, false);
    namedVector = conf.getBoolean(PartialVectorMerger.NAMED_VECTOR, false);
    logNormalize = conf.getBoolean(PartialVectorMerger.LOG_NORMALIZE, false);
  }

}
