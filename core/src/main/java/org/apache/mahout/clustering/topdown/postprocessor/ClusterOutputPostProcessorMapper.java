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
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Map;

/**
 * Mapper for post processing cluster output.
 */
public class ClusterOutputPostProcessorMapper extends
        Mapper<IntWritable, WeightedVectorWritable, IntWritable, VectorWritable> {

  private Map<Integer, Integer> newClusterMappings;
  private VectorWritable outputVector;

  //read the current cluster ids, and populate the cluster mapping hash table
  @Override
  public void setup(Context context) throws IOException {
    Configuration conf = context.getConfiguration();
    //this give the clusters-x-final directory where the cluster ids can be read
    Path clusterOutputPath = new Path(conf.get("clusterOutputPath"));
    //we want the key to be the cluster id, the value to be the index
    newClusterMappings = ClusterCountReader.getClusterIDs(clusterOutputPath, conf, true);
    outputVector = new VectorWritable();
  }

  @Override
  public void map(IntWritable key, WeightedVectorWritable val, Context context)
    throws IOException, InterruptedException {
    // by pivoting on the cluster mapping value, we can make sure that each unique cluster goes to it's own reducer,
    // since they are numbered from 0 to k-1, where k is the number of clusters
    outputVector.set(val.getVector());
    context.write(new IntWritable(newClusterMappings.get(key.get())), outputVector);
  }
}
