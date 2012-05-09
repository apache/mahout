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

import static org.apache.mahout.clustering.classify.ClusterClassificationConfigKeys.CLUSTERS_IN;
import static org.apache.mahout.clustering.classify.ClusterClassificationConfigKeys.EMIT_MOST_LIKELY;
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
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

/**
 * Mapper for classifying vectors into clusters.
 */
public class ClusterClassificationMapper extends
    Mapper<WritableComparable<?>,VectorWritable,IntWritable,WeightedVectorWritable> {
  
  private static double threshold;
  private List<Cluster> clusterModels;
  private ClusterClassifier clusterClassifier;
  private IntWritable clusterId;
  private WeightedVectorWritable weightedVW;
  private boolean emitMostLikely;
  
  @Override
  protected void setup(Context context) throws IOException,
      InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    String clustersIn = conf.get(CLUSTERS_IN);
    threshold = conf.getFloat(OUTLIER_REMOVAL_THRESHOLD, 0.0f);
    emitMostLikely = conf.getBoolean(EMIT_MOST_LIKELY, false);
    
    clusterModels = new ArrayList<Cluster>();
    
    if (clustersIn != null && !clustersIn.isEmpty()) {
      Path clustersInPath = new Path(clustersIn);
      clusterModels = populateClusterModels(clustersInPath, conf);
      ClusteringPolicy policy = ClusterClassifier
          .readPolicy(finalClustersPath(clustersInPath));
      clusterClassifier = new ClusterClassifier(clusterModels, policy);
    }
    clusterId = new IntWritable();
    weightedVW = new WeightedVectorWritable(1, null);
  }
  
  /**
   * Mapper which classifies the vectors to respective clusters.
   */
  @Override
  protected void map(WritableComparable<?> key, VectorWritable vw, Context context)
      throws IOException, InterruptedException {
    if (!clusterModels.isEmpty()) {
      Vector pdfPerCluster = clusterClassifier.classify(vw.get());
      if (shouldClassify(pdfPerCluster)) {
        if (emitMostLikely) {
          int maxValueIndex = pdfPerCluster.maxValueIndex();
          write(vw, context, maxValueIndex);
        } else {
          writeAllAboveThreshold(vw, context, pdfPerCluster);
        }
      }
    }
  }
  
  private void writeAllAboveThreshold(VectorWritable vw, Context context,
      Vector pdfPerCluster) throws IOException, InterruptedException {
    Iterator<Element> iterateNonZero = pdfPerCluster.iterateNonZero();
    while (iterateNonZero.hasNext()) {
      Element pdf = iterateNonZero.next();
      if (pdf.get() >= threshold) {
        int clusterIndex = pdf.index();
        write(vw, context, clusterIndex);
      }
    }
  }
  
  private void write(VectorWritable vw, Context context, int clusterIndex)
      throws IOException, InterruptedException {
    Cluster cluster = clusterModels.get(clusterIndex);
    clusterId.set(cluster.getId());
    weightedVW.setVector(vw.get());
    context.write(clusterId, weightedVW);
  }
  
  public static List<Cluster> populateClusterModels(Path clusterOutputPath, Configuration conf)
      throws IOException {
    List<Cluster> clusters = new ArrayList<Cluster>();
    Cluster cluster = null;
    FileSystem fileSystem = clusterOutputPath.getFileSystem(conf);
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath,
        PathFilters.finalPartFilter());
    Iterator<?> it = new SequenceFileDirValueIterator<Writable>(
        clusterFiles[0].getPath(), PathType.LIST, PathFilters.partFilter(),
        null, false, conf);
    while (it.hasNext()) {
      ClusterWritable next = (ClusterWritable) it.next();
      cluster = next.getValue();
      cluster.configure(conf);
      clusters.add(cluster);
    }
    return clusters;
  }
  
  private static boolean shouldClassify(Vector pdfPerCluster) {
    boolean isMaxPDFGreatherThanThreshold = pdfPerCluster.maxValue() >= threshold;
    return isMaxPDFGreatherThanThreshold;
  }
  
  private static Path finalClustersPath(Path clusterOutputPath)
      throws IOException {
    FileSystem fileSystem = clusterOutputPath
        .getFileSystem(new Configuration());
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath,
        PathFilters.finalPartFilter());
    Path finalClustersPath = clusterFiles[0].getPath();
    return finalClustersPath;
  }
}
