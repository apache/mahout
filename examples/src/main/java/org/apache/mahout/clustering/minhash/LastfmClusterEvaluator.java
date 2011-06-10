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

package org.apache.mahout.clustering.minhash;

import java.text.NumberFormat;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public final class LastfmClusterEvaluator {

  private LastfmClusterEvaluator() {
  }

  /* Calculate used JVM memory */
  private static String usedMemory() {
    Runtime runtime = Runtime.getRuntime();
    return "Used Memory: [" + (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024) + " MB] ";
  }

  /**
   * Computer Jaccard coefficient over two sets. (A intersect B) / (A union B)
   */
  private static double computeSimilarity(Iterable<Integer> listenerVector1, Iterable<Integer> listenerVector2) {
    Set<Integer> first = new HashSet<Integer>();
    for (Integer ele : listenerVector1) {
      first.add(ele);
    }
    Collection<Integer> second = new HashSet<Integer>();
    for (Integer ele : listenerVector2) {
      second.add(ele);
    }

    Collection<Integer> intersection = new HashSet<Integer>(first);
    intersection.retainAll(second);
    double intersectSize = intersection.size();

    first.addAll(second);
    double unionSize = first.size();
    return unionSize == 0 ? 0.0 : intersectSize / unionSize;
  }

  /**
   * Calculate the overall cluster precision by sampling clusters. Precision is
   * calculated as follows :-
   * 
   * 1. For a sample of all the clusters calculate the pair-wise similarity
   * (Jaccard coefficient) for items in the same cluster.
   * 
   * 2. Count true positives as items whose similarity is above specified
   * threshold.
   * 
   * 3. Precision = (true positives) / (total items in clusters sampled).
   * 
   * @param clusterFile
   *          The file containing cluster information
   * @param threshold
   *          Similarity threshold for containing two items in a cluster to be
   *          relevant. Must be between 0.0 and 1.0
   * @param samplePercentage
   *          Percentage of clusters to sample. Must be between 0.0 and 1.0
   */
  private static void testPrecision(Path clusterFile, double threshold, double samplePercentage) {
    Configuration conf = new Configuration();
    Random rand = RandomUtils.getRandom();
    Text prevCluster = new Text();
    List<List<Integer>> listenerVectors = Lists.newArrayList();
    long similarListeners = 0;
    long allListeners = 0;
    int clustersProcessed = 0;
    for (Pair<Text,VectorWritable> record :
         new SequenceFileIterable<Text,VectorWritable>(clusterFile, true, conf)) {
      Text cluster = record.getFirst();
      VectorWritable point = record.getSecond();
      if (!cluster.equals(prevCluster)) {
        // We got a new cluster
        prevCluster.set(cluster.toString());
        // Should we check previous cluster ?
        if (rand.nextDouble() > samplePercentage) {
          listenerVectors.clear();
          continue;
        }
        int numListeners = listenerVectors.size();
        allListeners += numListeners;
        for (int i = 0; i < numListeners; i++) {
          List<Integer> listenerVector1 = listenerVectors.get(i);
          for (int j = i + 1; j < numListeners; j++) {
            List<Integer> listenerVector2 = listenerVectors.get(j);
            double similarity = computeSimilarity(listenerVector1,
                listenerVector2);
            similarListeners += similarity >= threshold ? 1 : 0;
          }
        }
        listenerVectors.clear();
        clustersProcessed++;
        System.out.print('\r' + usedMemory() + " Clusters processed: " + clustersProcessed);
      }
      List<Integer> listeners = Lists.newArrayList();
      for (Vector.Element ele : point.get()) {
        listeners.add((int) ele.get());
      }
      listenerVectors.add(listeners);
    }
    System.out.println("\nTest Results");
    System.out.println("=============");
    System.out.println(" (A) Listeners in same cluster with simiarity above threshold ("
                           + threshold + ") : " + similarListeners);
    System.out.println(" (B) All listeners: " + allListeners);
    NumberFormat format = NumberFormat.getInstance();
    format.setMaximumFractionDigits(2);
    double precision = (double) similarListeners / allListeners * 100.0;
    System.out.println(" Average cluster precision: A/B = " + format.format(precision));
  }

  public static void main(String[] args) {
    if (args.length < 3) {
      System.out.println("LastfmClusterEvaluation <cluster-file> <threshold> <sample-percentage>");
      System.out.println("      <cluster-file>: Absolute Path of file containing cluster information in DEBUG format");
      System.out.println("         <threshold>: Minimum threshold for jaccard co-efficient for considering two items");
      System.out.println("                      in a cluster to be really similar. Should be between 0.0 and 1.0");
      System.out.println(" <sample-percentage>: Percentage of clusters to sample. Should be between 0.0 and 1.0");
      return;
    }
    Path clusterFile = new Path(args[0]);
    double threshold = Double.parseDouble(args[1]);
    double samplePercentage = Double.parseDouble(args[2]);
    testPrecision(clusterFile, threshold, samplePercentage);
  }

}
