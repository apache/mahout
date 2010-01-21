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
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FuzzyKMeansMapper extends MapReduceBase implements
    Mapper<WritableComparable<?>, VectorWritable, Text, FuzzyKMeansInfo> {

  private static final Logger log = LoggerFactory.getLogger(FuzzyKMeansMapper.class);

  private final List<SoftCluster> clusters = new ArrayList<SoftCluster>();
  private FuzzyKMeansClusterer clusterer; 
  
  @Override
  public void map(WritableComparable<?> key, VectorWritable point,
                  OutputCollector<Text, FuzzyKMeansInfo> output, Reporter reporter) throws IOException {
    clusterer.emitPointProbToCluster(point.get(), clusters, output);
  }

  /**
   * Configure the mapper by providing its clusters. Used by unit tests.
   *
   * @param clusters a List<Cluster>
   */
  void config(List<SoftCluster> clusters) {
    this.clusters.clear();
    this.clusters.addAll(clusters);
  }

  @Override
  public void configure(JobConf job) {

    super.configure(job);
    clusterer = new FuzzyKMeansClusterer(job);

    log.info("In Mapper Configure:");

    String clusterPath = job.get(FuzzyKMeansConfigKeys.CLUSTER_PATH_KEY);
    if (clusterPath != null && clusterPath.length() > 0) {
      FuzzyKMeansUtil.configureWithClusterInfo(clusterPath, clusters);
    }
    
    if (clusters.isEmpty()) {
      throw new NullPointerException("Cluster is empty!!!");
    }
  }

}
