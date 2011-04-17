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

package org.apache.mahout.clustering.spectral.eigencuts;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * <p>The point of this class is to take all the arrays of sensitivities
 * and convert them to a single matrix. Since there may be many values
 * that, according to their (i, j) coordinates, overlap in the matrix,
 * the "winner" will be determined by whichever value is smaller.</p> 
 */
public class EigencutsSensitivityReducer extends
    Reducer<IntWritable, EigencutsSensitivityNode, IntWritable, VectorWritable> {

  @Override
  protected void reduce(IntWritable key, Iterable<EigencutsSensitivityNode> arr, Context context)
    throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    Vector v = new RandomAccessSparseVector(conf.getInt(EigencutsKeys.AFFINITY_DIMENSIONS, Integer.MAX_VALUE), 100);
    double threshold = Double.parseDouble(conf.get(EigencutsKeys.TAU))
        / Double.parseDouble(conf.get(EigencutsKeys.DELTA));
    
    for (EigencutsSensitivityNode n : arr) {
      if (n.getSensitivity() < threshold && n.getSensitivity() < v.getQuick(n.getColumn())) {
        v.setQuick(n.getColumn(), n.getSensitivity());
      }
    }
    context.write(key, new VectorWritable(v));
  }
}
