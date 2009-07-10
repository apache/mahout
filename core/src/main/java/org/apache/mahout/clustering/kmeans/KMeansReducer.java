/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.kmeans;

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

public class KMeansReducer extends MapReduceBase implements
    Reducer<Text, KMeansInfo, Text, Cluster> {

  private Map<String, Cluster> clusterMap;

  @Override
  public void reduce(Text key, Iterator<KMeansInfo> values,
                     OutputCollector<Text, Cluster> output, Reporter reporter) throws IOException {
    Cluster cluster = clusterMap.get(key.toString());

    while (values.hasNext()) {
      KMeansInfo delta = values.next();
      cluster.addPoints(delta.getPoints(), delta.getPointTotal());
    }
    // force convergence calculation
    cluster.computeConvergence();
    output.collect(new Text(cluster.getIdentifier()), cluster);
  }

  @Override
  public void configure(JobConf job) {

    super.configure(job);
    Cluster.configure(job);
    clusterMap = new HashMap<String, Cluster>();

    List<Cluster> clusters = new ArrayList<Cluster>();
    KMeansUtil.configureWithClusterInfo(job.get(Cluster.CLUSTER_PATH_KEY),
        clusters);
    setClusterMap(clusters);

    if (clusterMap.isEmpty()) {
      throw new NullPointerException("Cluster is empty!!!");
    }
  }

  private void setClusterMap(List<Cluster> clusters) {
    clusterMap = new HashMap<String, Cluster>();
    for (Cluster cluster : clusters) {
      clusterMap.put(cluster.getIdentifier(), cluster);
    }
    clusters.clear();
  }

  public void config(List<Cluster> clusters) {
    setClusterMap(clusters);

  }

}
