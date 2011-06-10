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
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.math.VectorWritable;

public class FuzzyKMeansClusterMapper
    extends Mapper<WritableComparable<?>, VectorWritable, IntWritable, WeightedVectorWritable> {

  private final List<SoftCluster> clusters = Lists.newArrayList();

  private FuzzyKMeansClusterer clusterer;

  @Override
  protected void map(WritableComparable<?> key, VectorWritable point, Context context)
    throws IOException, InterruptedException {
    clusterer.emitPointToClusters(point, clusters, context);
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    clusterer = new FuzzyKMeansClusterer(conf);

    String clusterPath = conf.get(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY);
    if (clusterPath != null && clusterPath.length() > 0) {
      FuzzyKMeansUtil.configureWithClusterInfo(new Path(clusterPath), clusters);
    }

    if (clusters.isEmpty()) {
      throw new IllegalStateException("Cluster is empty!!!");
    }
  }

  /**
   * Configure the mapper by providing its clusters. Used by unit tests.
   * 
   * @param clusters
   *          a List<Cluster>
   */
  void setup(Collection<SoftCluster> clusters, Configuration conf) {
    this.clusters.clear();
    this.clusters.addAll(clusters);
    this.clusterer = new FuzzyKMeansClusterer(conf);
  }
}
