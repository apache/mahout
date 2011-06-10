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

import java.io.IOException;
import java.util.Collection;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.ClusterObservations;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.VectorWritable;

public class KMeansMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, ClusterObservations> {

  private KMeansClusterer clusterer;

  private final Collection<Cluster> clusters = Lists.newArrayList();

  @Override
  protected void map(WritableComparable<?> key, VectorWritable point, Context context)
    throws IOException, InterruptedException {
    this.clusterer.emitPointToNearestCluster(point.get(), this.clusters, context);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      DistanceMeasure measure = ccl.loadClass(conf.get(KMeansConfigKeys.DISTANCE_MEASURE_KEY))
          .asSubclass(DistanceMeasure.class).newInstance();
      measure.configure(conf);

      this.clusterer = new KMeansClusterer(measure);

      String clusterPath = conf.get(KMeansConfigKeys.CLUSTER_PATH_KEY);
      if (clusterPath != null && clusterPath.length() > 0) {
        KMeansUtil.configureWithClusterInfo(conf, new Path(clusterPath), clusters);
        if (clusters.isEmpty()) {
          throw new IllegalStateException("No clusters found. Check your -c path.");
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

  /**
   * Configure the mapper by providing its clusters. Used by unit tests.
   * 
   * @param clusters
   *          a List<Cluster>
   */
  void setup(Collection<Cluster> clusters, DistanceMeasure measure) {
    this.clusters.clear();
    this.clusters.addAll(clusters);
    this.clusterer = new KMeansClusterer(measure);
  }
}
