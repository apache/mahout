package org.apache.mahout.utils.vectors.io;
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


import com.google.common.collect.Lists;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Writer;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Implements a {@link org.apache.mahout.utils.vectors.io.ClusterWriter} that outputs in the format
 * used by ClusterDumper in Mahout 0.5
 */
public class ClusterDumperWriter extends AbstractClusterWriter implements ClusterWriter {
  private transient static Logger log = LoggerFactory.getLogger(ClusterDumperWriter.class);
  protected int subString;
  protected String[] dictionary;
  protected int numTopFeatures;

  public ClusterDumperWriter(Writer writer, Map<Integer, List<WeightedVectorWritable>> clusterIdToPoints, int numTopFeatures, String[] dictionary, int subString) {
    super(writer, clusterIdToPoints);
    this.numTopFeatures = numTopFeatures;
    this.dictionary = dictionary;
    this.subString = subString;
  }

  @Override
  public void write(Cluster value) throws IOException {
    String fmtStr = value.asFormatString(dictionary);
    if (subString > 0 && fmtStr.length() > subString) {
      writer.write(':');
      writer.write(fmtStr, 0, Math.min(subString, fmtStr.length()));
    } else {
      writer.write(fmtStr);
    }

    writer.write('\n');

    if (dictionary != null) {
      String topTerms = getTopFeatures(value.getCenter(), dictionary, numTopFeatures);
      writer.write("\tTop Terms: ");
      writer.write(topTerms);
      writer.write('\n');
    }

    List<WeightedVectorWritable> points = clusterIdToPoints.get(value.getId());
    if (points != null) {
      writer.write("\tWeight : [props - optional]:  Point:\n\t");
      for (Iterator<WeightedVectorWritable> iterator = points.iterator(); iterator.hasNext(); ) {
        WeightedVectorWritable point = iterator.next();
        writer.write(String.valueOf(point.getWeight()));
        if (point instanceof WeightedPropertyVectorWritable) {
          WeightedPropertyVectorWritable tmp = (WeightedPropertyVectorWritable) point;
          Map<Text, Text> map = tmp.getProperties();
          writer.write(" : [");
          for (Map.Entry<Text, Text> entry : map.entrySet()) {
            writer.write(entry.getKey().toString());
            writer.write("=");
            writer.write(entry.getValue().toString());
          }
          writer.write("]");
        }

        writer.write(": ");

        writer.write(AbstractCluster.formatVector(point.getVector(), dictionary));
        if (iterator.hasNext()) {
          writer.write("\n\t");
        }
      }
      writer.write('\n');
    }
  }
}
