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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Pattern;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.classify.WeightedVectorWritable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.StringUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;

/**
 * GraphML -- see http://gephi.org/users/supported-graph-formats/graphml-format/
 */
public class GraphMLClusterWriter extends AbstractClusterWriter {

  private static final Pattern VEC_PATTERN = Pattern.compile("\\{|\\:|\\,|\\}");
  private final Map<Integer, Color> colors = new HashMap<Integer, Color>();
  private Color lastClusterColor;
  private float lastX;
  private float lastY;
  private Random random;
  private int posStep;
  private final String[] dictionary;
  private final int numTopFeatures;
  private final int subString;

  public GraphMLClusterWriter(Writer writer, Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints,
                              DistanceMeasure measure, int numTopFeatures, String[] dictionary, int subString)
    throws IOException {
    super(writer, clusterIdToPoints, measure);
    this.dictionary = dictionary;
    this.numTopFeatures = numTopFeatures;
    this.subString = subString;
    init(writer);
  }

  private void init(Writer writer) throws IOException {
    writer.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    writer.append("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n"
                + "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
                + "xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\n"
                + "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">");
    //support rgb
    writer.append("<key attr.name=\"r\" attr.type=\"int\" for=\"node\" id=\"r\"/>\n"
                + "<key attr.name=\"g\" attr.type=\"int\" for=\"node\" id=\"g\"/>\n"
                + "<key attr.name=\"b\" attr.type=\"int\" for=\"node\" id=\"b\"/>"
                + "<key attr.name=\"size\" attr.type=\"int\" for=\"node\" id=\"size\"/>"
                + "<key attr.name=\"weight\" attr.type=\"float\" for=\"edge\" id=\"weight\"/>"
                + "<key attr.name=\"x\" attr.type=\"float\" for=\"node\" id=\"x\"/>"
                + "<key attr.name=\"y\" attr.type=\"float\" for=\"node\" id=\"y\"/>");
    writer.append("<graph edgedefault=\"undirected\">");
    lastClusterColor = new Color();
    posStep = (int) (0.1 * clusterIdToPoints.size()) + 100;
    random = RandomUtils.getRandom();
  }

  /*
    <?xml version="1.0" encoding="UTF-8"?>
    <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
    <graph id="G" edgedefault="undirected">
    <node id="n0"/>
    <node id="n1"/>
    <edge id="e1" source="n0" target="n1"/>
    </graph>
    </graphml>
   */

  @Override
  public void write(ClusterWritable clusterWritable) throws IOException {
    StringBuilder line = new StringBuilder();
    Cluster cluster = clusterWritable.getValue();
    Color rgb = getColor(cluster.getId());

    String topTerms = "";
    if (dictionary != null) {
      topTerms = getTopTerms(cluster.getCenter(), dictionary, numTopFeatures);
    }
    String clusterLabel = String.valueOf(cluster.getId()) + '_' + topTerms;
    //do some positioning so that items are visible and grouped together
    //TODO: put in a real layout algorithm
    float x = lastX + 1000;
    float y = lastY;
    if (x > (1000 + posStep)) {
      y = lastY + 1000;
      x = 0;
    }

    line.append(createNode(clusterLabel, rgb, x, y));
    List<WeightedPropertyVectorWritable> points = clusterIdToPoints.get(cluster.getId());
    if (points != null) {
      for (WeightedVectorWritable point : points) {
        Vector theVec = point.getVector();
        double distance = 1;
        if (measure != null) {
          //scale the distance
          distance = measure.distance(cluster.getCenter().getLengthSquared(), cluster.getCenter(), theVec) * 500;
        }
        String vecStr;
        int angle = random.nextInt(360); //pick an angle at random and then scale along that angle
        double angleRads = Math.toRadians(angle);

        float targetX = x + (float) (distance * Math.cos(angleRads));
        float targetY = y + (float) (distance * Math.sin(angleRads));
        if (theVec instanceof NamedVector) {
          vecStr = ((NamedVector) theVec).getName();
        } else {
          vecStr = theVec.asFormatString();
          //do some basic manipulations for display
          vecStr = VEC_PATTERN.matcher(vecStr).replaceAll("_");
        }
        if (subString > 0 && vecStr.length() > subString) {
          vecStr = vecStr.substring(0, subString);
        }
        line.append(createNode(vecStr, rgb, targetX, targetY));
        line.append(createEdge(clusterLabel, vecStr, distance));
      }
    }
    lastClusterColor = rgb;
    lastX = x;
    lastY = y;
    getWriter().append(line).append("\n");
  }

  private Color getColor(int clusterId) {
    Color result = colors.get(clusterId);
    if (result == null) {
      result = new Color();
      //there is probably some better way to color a graph
      int incR = 0;
      int incG = 0;
      int incB = 0;
      if (lastClusterColor.r + 20 < 256 && lastClusterColor.g + 20 < 256 && lastClusterColor.b + 20 < 256) {
        incR = 20;
        incG = 0;
        incB = 0;
      } else if (lastClusterColor.r + 20 >= 256 && lastClusterColor.g + 20 < 256 && lastClusterColor.b + 20 < 256) {
        incG = 20;
        incB = 0;
      } else if (lastClusterColor.r + 20 >= 256 && lastClusterColor.g + 20 >= 256 && lastClusterColor.b + 20 < 256) {
        incB = 20;
      } else {
        incR += 3;
        incG += 3;
        incR += 3;
      }
      result.r = (lastClusterColor.r + incR) % 256;
      result.g = (lastClusterColor.g + incG) % 256;
      result.b = (lastClusterColor.b + incB) % 256;
      colors.put(clusterId, result);
    }
    return result;
  }

  private static String createEdge(String left, String right, double distance) {
    left = StringUtils.escapeXML(left);
    right = StringUtils.escapeXML(right);
    return "<edge id=\"" + left + '_' + right + "\" source=\"" + left + "\" target=\"" + right + "\">" 
            + "<data key=\"weight\">" + distance + "</data></edge>";
  }

  private static String createNode(String s, Color rgb, float x, float y) {
    return "<node id=\"" + StringUtils.escapeXML(s) + "\"><data key=\"r\">" + rgb.r 
            + "</data>"
            + "<data key=\"g\">" + rgb.g
            + "</data>"
            + "<data key=\"b\">" + rgb.b
            + "</data>"
            + "<data key=\"x\">" + x
            + "</data>"
            + "<data key=\"y\">" + y
            + "</data>"
            + "</node>";
  }

  @Override
  public void close() throws IOException {
    getWriter().append("</graph>").append("</graphml>");
    super.close();
  }

  private static class Color {
    int r;
    int g;
    int b;
  }
}
