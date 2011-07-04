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

package org.apache.mahout.graph.common;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.graph.model.Triangle;
import org.apache.mahout.graph.model.UndirectedEdge;
import org.apache.mahout.graph.model.Vertex;
import org.apache.mahout.math.Varint;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Map;

/**
 * <p>Distributed computation of the local clustering coefficient of the vertices of an undirected graph. The local clustering coefficient is a
 * measure for the "connectedness" of a vertex in its neighborhood and is computed by dividing the number of closed triangles with a vertex'
 * neighbors by the number of possible triangles of this vertex with it's neighbours.</p>
 *
 * <p>The input files needs to be  {@link org.apache.hadoop.io.SequenceFile}s, one with {@link UndirectedEdge}s as keys and
 * any Writable as values, as it is already produced by {@link SimplifyGraphJob}, the other with {@link Triangle}s as keys and any Writable as
 * values, as it is already produced by {@link org.apache.mahout.graph.triangles.EnumerateTrianglesJob}</p>
 *
 * <p>This job outputs text files with a vertex id and it local clustering coefficient per line.</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--edges=(path): Directory containing one or more sequence files with edge data</li>
 * <li>--triangles=(path): Directory containing one or more sequence files with triangle data</li>
 * <li>--Dmapred.output.dir=(path): output path where the degree distribution data should be written</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public class LocalClusteringCoefficientJob extends AbstractJob {

  public static final String TMP_EDGES_PER_VERTEX = "edgesPerVertex";
  public static final String TMP_TRIANGLES_PER_VERTEX = "trianglesPerVertex";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new LocalClusteringCoefficientJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addOption("edges", "e", "path to the edges of the input graph", true);
    addOption("triangles", "t", "path to the triangles of the input graph", true);
    addOutputOption();

    Map<String, String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path edgesPath = new Path(parsedArgs.get("--edges"));
    Path trianglesPath = new Path(parsedArgs.get("--triangles"));

    // unfortunately we don't have access to an undeprecated MultipleInputs, so we need several M/R steps instead of one...
    Job countEdgesPerVertex = prepareJob(edgesPath, getTempPath(TMP_EDGES_PER_VERTEX),
        SequenceFileInputFormat.class, EdgeCountMapper.class, Vertex.class, TriangleOrEdgeCount.class, Reducer.class,
        Vertex.class, TriangleOrEdgeCount.class, SequenceFileOutputFormat.class);
    countEdgesPerVertex.setCombinerClass(TriangleOrEdgeCountCombiner.class);
    countEdgesPerVertex.waitForCompletion(true);

    Job countTrianglesPerVertex = prepareJob(trianglesPath, getTempPath(TMP_TRIANGLES_PER_VERTEX),
        SequenceFileInputFormat.class, TriangleCountMapper.class, Vertex.class, TriangleOrEdgeCount.class,
        Reducer.class, Vertex.class, TriangleOrEdgeCount.class, SequenceFileOutputFormat.class);
    countTrianglesPerVertex.setCombinerClass(TriangleOrEdgeCountCombiner.class);
    countTrianglesPerVertex.waitForCompletion(true);

    Job computeLocalClusteringCoefficient = prepareJob(getCombinedTempPath(TMP_EDGES_PER_VERTEX,
        TMP_TRIANGLES_PER_VERTEX), getOutputPath(), SequenceFileInputFormat.class, Mapper.class,
        Vertex.class, TriangleOrEdgeCount.class, LocalClusteringCoefficientReducer.class, LongWritable.class,
        DoubleWritable.class, TextOutputFormat.class);
    computeLocalClusteringCoefficient.setCombinerClass(TriangleOrEdgeCountCombiner.class);
    computeLocalClusteringCoefficient.waitForCompletion(true);

    return 0;
  }

  static class EdgeCountMapper extends Mapper<UndirectedEdge,Writable,Vertex,TriangleOrEdgeCount> {

    private static final TriangleOrEdgeCount ONE_EDGE = new TriangleOrEdgeCount(1, false);

    @Override
    protected void map(UndirectedEdge edge, Writable value, Context ctx) throws IOException, InterruptedException {
      ctx.write(edge.getFirstVertex(), ONE_EDGE);
      ctx.write(edge.getSecondVertex(), ONE_EDGE);
    }
  }

  static class TriangleCountMapper extends Mapper<Triangle,Writable,Vertex,TriangleOrEdgeCount> {

    private static final TriangleOrEdgeCount ONE_TRIANGLE = new TriangleOrEdgeCount(1, true);

    @Override
    protected void map(Triangle triangle, Writable value, Context ctx) throws IOException, InterruptedException {
      ctx.write(triangle.getFirstVertex(), ONE_TRIANGLE);
      ctx.write(triangle.getSecondVertex(), ONE_TRIANGLE);
      ctx.write(triangle.getThirdVertex(), ONE_TRIANGLE);
    }
  }

  static class LocalClusteringCoefficientReducer
      extends Reducer<Vertex,TriangleOrEdgeCount,LongWritable,DoubleWritable> {
    @Override
    protected void reduce(Vertex vertex, Iterable<TriangleOrEdgeCount> counts, Context ctx)
        throws IOException, InterruptedException {
      int numEdges = 0;
      int numTriangles = 0;

      for (TriangleOrEdgeCount count : counts) {
        if (count.isTriangles()) {
          numTriangles += count.get();
        } else {
          numEdges += count.get();
        }
      }

      double localClusteringCoefficient = numEdges > 1 ?
          (double) numTriangles / (double) (numEdges * (numEdges - 1)) : 0.0;

      ctx.write(new LongWritable(vertex.getId()), new DoubleWritable(localClusteringCoefficient));
    }
  }

  static class TriangleOrEdgeCountCombiner extends Reducer<Vertex,TriangleOrEdgeCount,Vertex,TriangleOrEdgeCount> {

    @Override
    protected void reduce(Vertex vertex, Iterable<TriangleOrEdgeCount> counts, Context ctx)
        throws IOException, InterruptedException {
      int numEdges = 0;
      int numTriangles = 0;

      for (TriangleOrEdgeCount count : counts) {
        if (count.isTriangles()) {
          numTriangles += count.get();
        } else {
          numEdges += count.get();
        }
      }

      if (numEdges > 0) {
        ctx.write(vertex, new TriangleOrEdgeCount(numEdges, false));
      }
      if (numTriangles > 0) {
        ctx.write(vertex, new TriangleOrEdgeCount(numTriangles, true));
      }
    }
  }


  static class TriangleOrEdgeCount implements Writable {

    private int count;
    private boolean isTriangles;

    TriangleOrEdgeCount() {}

    public int get() {
      return count;
    }

    public boolean isTriangles() {
      return isTriangles;
    }

    TriangleOrEdgeCount(int count, boolean isTriangle) {
      this.count = count;
      this.isTriangles = isTriangle;
    }

    @Override
    public void write(DataOutput out) throws IOException {
      Varint.writeUnsignedVarInt(count, out);
      out.writeBoolean(isTriangles);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
      count = Varint.readUnsignedVarInt(in);
      isTriangles = in.readBoolean();
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof TriangleOrEdgeCount) {
        TriangleOrEdgeCount other = (TriangleOrEdgeCount) o;
        return count == other.count && isTriangles == other.isTriangles;
      }
      return false;
    }

    @Override
    public int hashCode() {
      return 31 * count + (isTriangles ? 1 : 0);
    }
  }
}
