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
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.distance.DistanceMeasure;

public class KMeansReducer extends MapReduceBase implements Reducer<Text,KMeansInfo,Text,Cluster> {
  
  private Map<String,Cluster> clusterMap;
  private double convergenceDelta;
  private DistanceMeasure measure;
  
  @Override
  public void reduce(Text key,
                     Iterator<KMeansInfo> values,
                     OutputCollector<Text,Cluster> output,
                     Reporter reporter) throws IOException {
    Cluster cluster = clusterMap.get(key.toString());
    
    while (values.hasNext()) {
      KMeansInfo delta = values.next();
      cluster.addPoints(delta.getPoints(), delta.getPointTotal());
    }
    // force convergence calculation
    boolean converged = cluster.computeConvergence(this.measure, this.convergenceDelta);
    if (converged) {
      reporter.incrCounter("Clustering", "Converged Clusters", 1);
    }
    output.collect(new Text(cluster.getIdentifier()), cluster);
  }
  
  @Override
  public void configure(JobConf job) {
    
    super.configure(job);
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl = ccl.loadClass(job.get(KMeansConfigKeys.DISTANCE_MEASURE_KEY));
      this.measure = (DistanceMeasure) cl.newInstance();
      this.measure.configure(job);
      
      this.convergenceDelta = Double.parseDouble(job.get(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY));
      
      this.clusterMap = new HashMap<String,Cluster>();
      
      String path = job.get(KMeansConfigKeys.CLUSTER_PATH_KEY);
      if (path.length() > 0) {
        List<Cluster> clusters = new ArrayList<Cluster>();
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
  
  private void setClusterMap(List<Cluster> clusters) {
    clusterMap = new HashMap<String,Cluster>();
    for (Cluster cluster : clusters) {
      clusterMap.put(cluster.getIdentifier(), cluster);
    }
    clusters.clear();
  }
  
  public void config(List<Cluster> clusters) {
    setClusterMap(clusters);
    
  }
  
}
