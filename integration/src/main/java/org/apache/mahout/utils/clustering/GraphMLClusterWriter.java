package org.apache.mahout.utils.clustering;


import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.vectors.io.AbstractClusterWriter;
import org.apache.mahout.utils.vectors.io.ClusterWriter;

import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.Map;

/**
 * GraphML -- see http://gephi.org/users/supported-graph-formats/graphml-format/
 *
 **/
public class GraphMLClusterWriter extends AbstractClusterWriter implements ClusterWriter {

  public GraphMLClusterWriter(Writer writer, Map<Integer, List<WeightedVectorWritable>> clusterIdToPoints) throws IOException {
    super(writer, clusterIdToPoints);
    writer.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>");
    writer.append("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n" +
            "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n" +
            "xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns\n" +
            "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">");
    writer.append("<graph edgedefault=\"undirected\">");
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
  public void write(Cluster cluster) throws IOException {
    StringBuilder line = new StringBuilder();
    line.append(createNode(String.valueOf(cluster.getId())));
    List<WeightedVectorWritable> points = clusterIdToPoints.get(cluster.getId());
    if (points != null) {
      for (WeightedVectorWritable point : points) {
        Vector theVec = point.getVector();
        String vecStr;
        if (theVec instanceof NamedVector){
          vecStr = ((NamedVector)theVec).getName();
          line.append(createNode(vecStr));
        } else {
          vecStr = theVec.asFormatString();
          //do some basic manipulations for display
          vecStr = vecStr.replaceAll("\\{|\\:|\\,|\\}", "_");
          line.append(createNode(vecStr));
        }
        line.append(createEdge(String.valueOf(cluster.getId()), vecStr));
      }
      writer.append(line).append("\n");
    }
  }

  private String createEdge(String left, String right) {
    return "<edge id=\"" + left + "_" + right + "\" source=\"" + left + "\" target=\"" + right + "\"/>";
  }

  private String createNode(String s) {
    return "<node id=\"" + s + "\"/>";
  }

  @Override
  public void close() throws IOException {
    writer.append("</graph>");
    super.close();
  }
}
