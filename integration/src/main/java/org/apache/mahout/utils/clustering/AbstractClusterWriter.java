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

package org.apache.mahout.utils.clustering;

import java.io.IOException;
import java.io.Writer;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

/**
 * Base class for implementing ClusterWriter
 */
public abstract class AbstractClusterWriter implements ClusterWriter {

  private static final Logger log = LoggerFactory.getLogger(AbstractClusterWriter.class);

  protected final Writer writer;
  protected final Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints;
  protected final DistanceMeasure measure;

  /**
   *
   * @param writer The underlying {@link java.io.Writer} to use
   * @param clusterIdToPoints The map between cluster ids {@link org.apache.mahout.clustering.Cluster#getId()} and the
   *                          points in the cluster
   * @param measure The {@link org.apache.mahout.common.distance.DistanceMeasure} used to calculate the distance.
   *                Some writers may wish to use it for calculating weights for display.  May be null.
   */
  protected AbstractClusterWriter(Writer writer, Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints,
      DistanceMeasure measure) {
    this.writer = writer;
    this.clusterIdToPoints = clusterIdToPoints;
    this.measure = measure;
  }

  protected Writer getWriter() {
    return writer;
  }

  protected Map<Integer, List<WeightedPropertyVectorWritable>> getClusterIdToPoints() {
    return clusterIdToPoints;
  }

  public static String getTopFeatures(Vector vector, String[] dictionary, int numTerms) {

    StringBuilder sb = new StringBuilder(100);

    for (Pair<String, Double> item : getTopPairs(vector, dictionary, numTerms)) {
      String term = item.getFirst();
      sb.append("\n\t\t");
      sb.append(StringUtils.rightPad(term, 40));
      sb.append("=>");
      sb.append(StringUtils.leftPad(item.getSecond().toString(), 20));
    }
    return sb.toString();
  }

  public static String getTopTerms(Vector vector, String[] dictionary, int numTerms) {

    StringBuilder sb = new StringBuilder(100);

    for (Pair<String, Double> item : getTopPairs(vector, dictionary, numTerms)) {
      String term = item.getFirst();
      sb.append(term).append('_');
    }
    sb.deleteCharAt(sb.length() - 1);
    return sb.toString();
  }

  @Override
  public long write(Iterable<ClusterWritable> iterable) throws IOException {
    return write(iterable, Long.MAX_VALUE);
  }

  @Override
  public void close() throws IOException {
    writer.close();
  }

  @Override
  public long write(Iterable<ClusterWritable> iterable, long maxDocs) throws IOException {
    long result = 0;
    Iterator<ClusterWritable> iterator = iterable.iterator();
    while (result < maxDocs && iterator.hasNext()) {
      write(iterator.next());
      result++;
    }
    return result;
  }

  private static Collection<Pair<String, Double>> getTopPairs(Vector vector, String[] dictionary, int numTerms) {
    List<TermIndexWeight> vectorTerms = Lists.newArrayList();

    for (Vector.Element elt : vector.nonZeroes()) {
      vectorTerms.add(new TermIndexWeight(elt.index(), elt.get()));
    }

    // Sort results in reverse order (ie weight in descending order)
    Collections.sort(vectorTerms, new Comparator<TermIndexWeight>() {
      @Override
      public int compare(TermIndexWeight one, TermIndexWeight two) {
        return Double.compare(two.weight, one.weight);
      }
    });

    Collection<Pair<String, Double>> topTerms = Lists.newLinkedList();

    for (int i = 0; i < vectorTerms.size() && i < numTerms; i++) {
      int index = vectorTerms.get(i).index;
      String dictTerm = dictionary[index];
      if (dictTerm == null) {
        log.error("Dictionary entry missing for {}", index);
        continue;
      }
      topTerms.add(new Pair<String, Double>(dictTerm, vectorTerms.get(i).weight));
    }

    return topTerms;
  }

  private static class TermIndexWeight {
    private final int index;
    private final double weight;

    TermIndexWeight(int index, double weight) {
      this.index = index;
      this.weight = weight;
    }
  }
}
