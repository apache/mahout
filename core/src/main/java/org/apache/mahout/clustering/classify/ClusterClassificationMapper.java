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

package org.apache.mahout.clustering.classify;

import static org.apache.mahout.clustering.classify.ClusterClassificationConfigKeys.OUTLIER_REMOVAL_THRESHOLD;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

/**
 * Mapper for classifying vectors into clusters.
 */
public class ClusterClassificationMapper extends Mapper<IntWritable,VectorWritable,IntWritable,WeightedVectorWritable> {
  
  private static double threshold;
  private List<Cluster> clusterModels;
  private ClusterClassifier clusterClassifier;
  private IntWritable clusterId;
  private WeightedVectorWritable weightedVW;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    String clustersIn = conf.get(ClusterClassificationConfigKeys.CLUSTERS_IN);
    
    clusterModels = new ArrayList<Cluster>();
    
    if (clustersIn != null && !clustersIn.isEmpty()) {
      Path clustersInPath = new Path(clustersIn, "*");
      populateClusterModels(clustersInPath);
      ClusteringPolicy policy = ClusterClassifier.readPolicy(clustersInPath);
      clusterClassifier = new ClusterClassifier(clusterModels, policy);
    }
    threshold = conf.getFloat(OUTLIER_REMOVAL_THRESHOLD, 0.0f);
    clusterId = new IntWritable();
    weightedVW = new WeightedVectorWritable(1, null);
  }
  
  @Override
  protected void map(IntWritable key, VectorWritable vw, Context context) throws IOException, InterruptedException {
    if (!clusterModels.isEmpty()) {
      Vector pdfPerCluster = clusterClassifier.classify(vw.get());
      if (shouldClassify(pdfPerCluster)) {
        int maxValueIndex = pdfPerCluster.maxValueIndex();
        Cluster cluster = clusterModels.get(maxValueIndex);
        clusterId.set(cluster.getId());
        weightedVW.setVector(vw.get());
        context.write(clusterId, weightedVW);
      }
    }
  }
  
  public static List<Cluster> populateClusterModels(Path clusterOutputPath) throws IOException {
    List<Cluster> clusterModels = new ArrayList<Cluster>();
    Configuration conf = new Configuration();
    Cluster cluster = null;
    FileSystem fileSystem = clusterOutputPath.getFileSystem(conf);
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());
    Iterator<?> it = new SequenceFileDirValueIterator<Writable>(clusterFiles[0].getPath(), PathType.LIST,
        PathFilters.partFilter(), null, false, conf);
    while (it.hasNext()) {
      cluster = (Cluster) it.next();
      clusterModels.add(cluster);
    }
    return clusterModels;
  }
  
  private static boolean shouldClassify(Vector pdfPerCluster) {
    return pdfPerCluster.maxValue() >= threshold;
  }
  
}
