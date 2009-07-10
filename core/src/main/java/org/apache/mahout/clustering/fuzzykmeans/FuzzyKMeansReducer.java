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

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class FuzzyKMeansReducer extends MapReduceBase implements
    Reducer<Text, FuzzyKMeansInfo, Text, SoftCluster> {

  protected Map<String, SoftCluster> clusterMap;

  @Override
  public void reduce(Text key, Iterator<FuzzyKMeansInfo> values,
                     OutputCollector<Text, SoftCluster> output, Reporter reporter) throws IOException {

    SoftCluster cluster = clusterMap.get(key.toString());

    while (values.hasNext()) {
      FuzzyKMeansInfo value = values.next();

      if (value.combinerPass == 0) // escaped from combiner
      {
        cluster.addPoint(value.getVector(), Math.pow(value.getProbability(), SoftCluster.getM()));
      } else {
        cluster.addPoints(value.getVector(), value.getProbability());
      }

    }
    // force convergence calculation
    cluster.computeConvergence();
    output.collect(new Text(cluster.getIdentifier()), cluster);
  }

  @Override
  public void configure(JobConf job) {

    super.configure(job);
    SoftCluster.configure(job);
    clusterMap = new HashMap<String, SoftCluster>();

    List<SoftCluster> clusters = new ArrayList<SoftCluster>();
    FuzzyKMeansUtil.configureWithClusterInfo(job
        .get(SoftCluster.CLUSTER_PATH_KEY), clusters);
    setClusterMap(clusters);

    if (clusterMap.isEmpty()) {
      throw new NullPointerException("Cluster is empty!!!");
    }
  }

  private void setClusterMap(List<SoftCluster> clusters) {
    clusterMap = new HashMap<String, SoftCluster>();
    for (SoftCluster cluster : clusters) {
      clusterMap.put(cluster.getIdentifier(), cluster);
    }
    clusters.clear();
  }

  public void config(List<SoftCluster> clusters) {
    setClusterMap(clusters);

  }

}
