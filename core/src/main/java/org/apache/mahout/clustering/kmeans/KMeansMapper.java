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
package org.apache.mahout.clustering.kmeans;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KMeansMapper extends MapReduceBase implements
    Mapper<WritableComparable<?>, VectorWritable, Text, KMeansInfo> {

  private KMeansClusterer clusterer;
  private final List<Cluster> clusters = new ArrayList<Cluster>();

  @Override
  public void map(WritableComparable<?> key, VectorWritable point,
      OutputCollector<Text, KMeansInfo> output, Reporter reporter)
      throws IOException {
   this.clusterer.emitPointToNearestCluster(point.get(), this.clusters, output);
  }

  /**
   * Configure the mapper by providing its clusters. Used by unit tests.
   * 
   * @param clusters
   *          a List<Cluster>
   */
  void config(List<Cluster> clusters) {
    this.clusters.clear();
    this.clusters.addAll(clusters);
  }

  @Override
  public void configure(JobConf job) {
    super.configure(job);
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl = ccl.loadClass(job
          .get(KMeansConfigKeys.DISTANCE_MEASURE_KEY));
      DistanceMeasure measure = (DistanceMeasure) cl.newInstance();
      measure.configure(job);

      this.clusterer = new KMeansClusterer(measure);

      String clusterPath = job.get(KMeansConfigKeys.CLUSTER_PATH_KEY);
      if (clusterPath != null && clusterPath.length() > 0) {
        KMeansUtil.configureWithClusterInfo(clusterPath, clusters);
        if (clusters.isEmpty()) {
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
}
