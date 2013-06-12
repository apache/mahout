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

package org.apache.mahout.clustering.topdown.postprocessor;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Map;

/**
 * Reducer for post processing cluster output.
 */
public class ClusterOutputPostProcessorReducer
    extends Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {

  private Map<Integer, Integer> reverseClusterMappings;

  //read the current cluster ids, and populate the hash cluster mapping hash table
  @Override
  public void setup(Context context) throws IOException {
    Configuration conf = context.getConfiguration();
    Path clusterOutputPath = new Path(conf.get("clusterOutputPath"));
    //we want to the key to be the index, the value to be the cluster id
    reverseClusterMappings = ClusterCountReader.getClusterIDs(clusterOutputPath, conf, false);
  }

  /**
   * The key is the remapped cluster id and the values contains the vectors in that cluster.
   */
  @Override
  protected void reduce(IntWritable key, Iterable<VectorWritable> values, Context context) throws IOException,
          InterruptedException {
    //remap the cluster back to its original id
    //and then output the vectors with their correct
    //cluster id.
    IntWritable outKey = new IntWritable(reverseClusterMappings.get(key.get()));
    System.out.println(outKey + " this: " + this);
    for (VectorWritable value : values) {
      context.write(outKey, value);
    }
  }

}
