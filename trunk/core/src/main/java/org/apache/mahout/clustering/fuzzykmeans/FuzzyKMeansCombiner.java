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

package org.apache.mahout.clustering.fuzzykmeans;

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.ClusterObservations;

public class FuzzyKMeansCombiner extends Reducer<Text, ClusterObservations, Text, ClusterObservations> {

  private FuzzyKMeansClusterer clusterer;

  @Override
  protected void reduce(Text key, Iterable<ClusterObservations> values, Context context)
    throws IOException, InterruptedException {
    SoftCluster cluster = new SoftCluster();
    for (ClusterObservations value : values) {
      if (value.getCombinerState() == 0) { // first time through combiner
        cluster.observe(value.getS1(), Math.pow(value.getS0(), clusterer.getM()));
      } else {
        cluster.observe(value);
      }
    }
    context.write(key, cluster.getObservations().incrementCombinerState());
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    clusterer = new FuzzyKMeansClusterer(context.getConfiguration());
  }
}
