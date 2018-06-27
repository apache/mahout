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

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Format is adjacency style as put forth at http://gephi.org/users/supported-graph-formats/csv-format/, the centroid
 * is the first element and all the rest of the row are the points in that cluster
 *
 **/
public class CSVClusterWriter extends AbstractClusterWriter {

  private static final Pattern VEC_PATTERN = Pattern.compile("\\{|\\:|\\,|\\}");

  public CSVClusterWriter(Writer writer, Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints,
      DistanceMeasure measure) {
    super(writer, clusterIdToPoints, measure);
  }

  @Override
  public void write(ClusterWritable clusterWritable) throws IOException {
    StringBuilder line = new StringBuilder();
    Cluster cluster = clusterWritable.getValue();
    line.append(cluster.getId());
    List<WeightedPropertyVectorWritable> points = getClusterIdToPoints().get(cluster.getId());
    if (points != null) {
      for (WeightedPropertyVectorWritable point : points) {
        Vector theVec = point.getVector();
        line.append(',');
        if (theVec instanceof NamedVector) {
          line.append(((NamedVector)theVec).getName());
        } else {
          String vecStr = theVec.asFormatString();
          //do some basic manipulations for display
          vecStr = VEC_PATTERN.matcher(vecStr).replaceAll("_");
          line.append(vecStr);
        }
      }
      getWriter().append(line).append("\n");
    }
  }
}
