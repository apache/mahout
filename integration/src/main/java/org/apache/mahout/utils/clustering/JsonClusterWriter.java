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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import com.google.common.collect.Maps;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.codehaus.jackson.map.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

/**
 * Dump cluster info to JSON formatted lines. Heavily inspired by
 * ClusterDumperWriter.java and CSVClusterWriter.java
 *
 */
public class JsonClusterWriter extends AbstractClusterWriter {
  private final String[] dictionary;
  private final int numTopFeatures;
  private final ObjectMapper jxn;

  private static final Logger log = LoggerFactory
      .getLogger(JsonClusterWriter.class);
  private static final Pattern VEC_PATTERN = Pattern.compile("\\{|\\:|\\,|\\}");

  public JsonClusterWriter(Writer writer,
      Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints,
      DistanceMeasure measure, int numTopFeatures, String[] dictionary) {
    super(writer, clusterIdToPoints, measure);
    this.numTopFeatures = numTopFeatures;
    this.dictionary = dictionary;
    jxn = new ObjectMapper();
  }

  /**
   * Generate HashMap with cluster info and write as a single JSON formatted
   * line
   */
  @Override
  public void write(ClusterWritable clusterWritable) throws IOException {
    Map<String, Object> res = Maps.newHashMap();

    // get top terms
    List<Object> topTerms = getTopFeaturesList(clusterWritable.getValue()
        .getCenter(), dictionary, numTopFeatures);
    res.put("top_terms", topTerms);

    // get human-readable cluster representation
    Cluster cluster = clusterWritable.getValue();
    String fmtStr = cluster.asFormatString(dictionary);
    res.put("cluster_id", cluster.getId());
    res.put("cluster", fmtStr);

    // get points
    List<Object> points = getPoints(cluster, dictionary);
    res.put("points", points);

    // write JSON
    Writer writer = getWriter();
    writer.write(jxn.writeValueAsString(res) + "\n");
  }

  /**
   * Create a List of HashMaps containing top terms information
   *
   * @return List<Object>
   */
  public List<Object> getTopFeaturesList(Vector vector, String[] dictionary,
      int numTerms) {

    List<TermIndexWeight> vectorTerms = Lists.newArrayList();

    for (Vector.Element elt : vector.nonZeroes()) {
      vectorTerms.add(new TermIndexWeight(elt.index(), elt.get()));
    }

    // Sort results in reverse order (i.e. weight in descending order)
    Collections.sort(vectorTerms, new Comparator<TermIndexWeight>() {
      @Override
      public int compare(TermIndexWeight one, TermIndexWeight two) {
        return Double.compare(two.weight, one.weight);
      }
    });

    List<Object> topTerms = Lists.newLinkedList();

    for (int i = 0; i < vectorTerms.size() && i < numTerms; i++) {
      int index = vectorTerms.get(i).index;
      String dictTerm = dictionary[index];
      if (dictTerm == null) {
        log.error("Dictionary entry missing for {}", index);
        continue;
      }
      Map<String, Object> term_entry = Maps.newHashMap();
      term_entry.put("term", dictTerm);
      term_entry.put("weight", vectorTerms.get(i).weight);
      topTerms.add(term_entry);
    }

    return topTerms;
  }

  /**
   * Create a List of HashMaps containing Vector point information
   *
   * @return List<Object>
   */
  public List<Object> getPoints(Cluster cluster, String[] dictionary) {
    List<Object> vectorObjs = Lists.newLinkedList();
    List<WeightedPropertyVectorWritable> points = getClusterIdToPoints().get(
        cluster.getId());

    if (points != null) {
      for (WeightedPropertyVectorWritable point : points) {
        Map<String, Object> entry = Maps.newHashMap();
        Vector theVec = point.getVector();
        if (theVec instanceof NamedVector) {
          entry.put("vector_name", ((NamedVector) theVec).getName());
        } else {
          String vecStr = theVec.asFormatString();
          // do some basic manipulations for display
          vecStr = VEC_PATTERN.matcher(vecStr).replaceAll("_");
          entry.put("vector_name", vecStr);
        }
        entry.put("weight", String.valueOf(point.getWeight()));
        entry.put("point",
            AbstractCluster.formatVector(point.getVector(), dictionary));
        vectorObjs.add(entry);
      }
    }
    return vectorObjs;
  }

  /**
   * Convenience class for sorting terms
   *
   */
  private static class TermIndexWeight {
    private final int index;
    private final double weight;

    TermIndexWeight(int index, double weight) {
      this.index = index;
      this.weight = weight;
    }
  }

}
