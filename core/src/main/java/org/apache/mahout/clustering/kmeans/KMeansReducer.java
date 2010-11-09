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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.ClusterObservations;
import org.apache.mahout.common.distance.DistanceMeasure;

public class KMeansReducer extends Reducer<Text, ClusterObservations, Text, Cluster> {

  private Map<String, Cluster> clusterMap;

  private KMeansClusterer clusterer;

  @Override
  protected void reduce(Text key, Iterable<ClusterObservations> values, Context context) throws IOException, InterruptedException {
    Cluster cluster = clusterMap.get(key.toString());
    for (ClusterObservations delta : values) {
      cluster.observe(delta);
    }
    // force convergence calculation
    boolean converged = clusterer.computeConvergence(cluster);
    if (converged) {
      context.getCounter("Clustering", "Converged Clusters").increment(1);
    }
    cluster.computeParameters();
    context.write(new Text(cluster.getIdentifier()), cluster);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    try {
      this.clusterer = new KMeansClusterer(conf);
      this.clusterMap = new HashMap<String, Cluster>();

      String path = conf.get(KMeansConfigKeys.CLUSTER_PATH_KEY);
      if (path.length() > 0) {
        Collection<Cluster> clusters = new ArrayList<Cluster>();
        KMeansUtil.configureWithClusterInfo(new Path(path), clusters);
        setClusterMap(clusters);
        if (clusterMap.isEmpty()) {
          throw new IllegalStateException("Cluster is empty!");
        }
      }
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
  }

  private void setClusterMap(Collection<Cluster> clusters) {
    clusterMap = new HashMap<String, Cluster>();
    for (Cluster cluster : clusters) {
      clusterMap.put(cluster.getIdentifier(), cluster);
    }
    clusters.clear();
  }

  public void setup(Collection<Cluster> clusters, DistanceMeasure measure) {
    setClusterMap(clusters);
    this.clusterer = new KMeansClusterer(measure);
  }

}
