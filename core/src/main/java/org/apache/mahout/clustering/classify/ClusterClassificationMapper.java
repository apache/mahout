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

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.iterator.DistanceMeasureCluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

/**
 * Mapper for classifying vectors into clusters.
 */
public class ClusterClassificationMapper extends
    Mapper<WritableComparable<?>,VectorWritable,IntWritable,WeightedVectorWritable> {
  
  private double threshold;
  private List<Cluster> clusterModels;
  private ClusterClassifier clusterClassifier;
  private IntWritable clusterId;
  private boolean emitMostLikely;
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    String clustersIn = conf.get(ClusterClassificationConfigKeys.CLUSTERS_IN);
    threshold = conf.getFloat(ClusterClassificationConfigKeys.OUTLIER_REMOVAL_THRESHOLD, 0.0f);
    emitMostLikely = conf.getBoolean(ClusterClassificationConfigKeys.EMIT_MOST_LIKELY, false);
    
    clusterModels = Lists.newArrayList();
    
    if (clustersIn != null && !clustersIn.isEmpty()) {
      Path clustersInPath = new Path(clustersIn);
      clusterModels = populateClusterModels(clustersInPath, conf);
      ClusteringPolicy policy = ClusterClassifier
          .readPolicy(finalClustersPath(clustersInPath));
      clusterClassifier = new ClusterClassifier(clusterModels, policy);
    }
    clusterId = new IntWritable();
  }
  
  /**
   * Mapper which classifies the vectors to respective clusters.
   */
  @Override
  protected void map(WritableComparable<?> key, VectorWritable vw, Context context)
    throws IOException, InterruptedException {
    if (!clusterModels.isEmpty()) {
      // Converting to NamedVectors to preserve the vectorId else its not obvious as to which point
      // belongs to which cluster - fix for MAHOUT-1410
      Class<? extends Vector> vectorClass = vw.get().getClass();
      Vector vector = vw.get();
      if (!vectorClass.equals(NamedVector.class)) {
        if (key.getClass().equals(Text.class)) {
          vector = new NamedVector(vector, key.toString());
        } else if (key.getClass().equals(IntWritable.class)) {
          vector = new NamedVector(vector, Integer.toString(((IntWritable) key).get()));
        }
      }
      Vector pdfPerCluster = clusterClassifier.classify(vector);
      if (shouldClassify(pdfPerCluster)) {
        if (emitMostLikely) {
          int maxValueIndex = pdfPerCluster.maxValueIndex();
          write(new VectorWritable(vector), context, maxValueIndex, 1.0);
        } else {
          writeAllAboveThreshold(new VectorWritable(vector), context, pdfPerCluster);
        }
      }
    }
  }
  
  private void writeAllAboveThreshold(VectorWritable vw, Context context,
      Vector pdfPerCluster) throws IOException, InterruptedException {
    for (Element pdf : pdfPerCluster.nonZeroes()) {
      if (pdf.get() >= threshold) {
        int clusterIndex = pdf.index();
        write(vw, context, clusterIndex, pdf.get());
      }
    }
  }
  
  private void write(VectorWritable vw, Context context, int clusterIndex, double weight)
    throws IOException, InterruptedException {
    Cluster cluster = clusterModels.get(clusterIndex);
    clusterId.set(cluster.getId());

    DistanceMeasureCluster distanceMeasureCluster = (DistanceMeasureCluster) cluster;
    DistanceMeasure distanceMeasure = distanceMeasureCluster.getMeasure();
    double distance = distanceMeasure.distance(cluster.getCenter(), vw.get());

    Map<Text, Text> props = Maps.newHashMap();
    props.put(new Text("distance"), new Text(Double.toString(distance)));
    context.write(clusterId, new WeightedPropertyVectorWritable(weight, vw.get(), props));
  }
  
  public static List<Cluster> populateClusterModels(Path clusterOutputPath, Configuration conf) throws IOException {
    List<Cluster> clusters = Lists.newArrayList();
    FileSystem fileSystem = clusterOutputPath.getFileSystem(conf);
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());
    Iterator<?> it = new SequenceFileDirValueIterator<Writable>(
        clusterFiles[0].getPath(), PathType.LIST, PathFilters.partFilter(),
        null, false, conf);
    while (it.hasNext()) {
      ClusterWritable next = (ClusterWritable) it.next();
      Cluster cluster = next.getValue();
      cluster.configure(conf);
      clusters.add(cluster);
    }
    return clusters;
  }
  
  private boolean shouldClassify(Vector pdfPerCluster) {
    return pdfPerCluster.maxValue() >= threshold;
  }
  
  private static Path finalClustersPath(Path clusterOutputPath) throws IOException {
    FileSystem fileSystem = clusterOutputPath.getFileSystem(new Configuration());
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());
    return clusterFiles[0].getPath();
  }
}
