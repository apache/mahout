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

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.graph.model.Triangle;
import org.apache.mahout.graph.model.UndirectedEdge;
import org.apache.mahout.graph.model.Vertex;

/** Enumerates all triangles of an undirected graph. */
public class EnumerateTrianglesJob extends AbstractJob {

  public static final String TMP_AUGMENTED_EDGES = "augmentedEdges";
  public static final String TMP_EDGES_WITH_DEGREES = "edgesWithDegrees";
  public static final String TMP_CLOSING_EDGES = "closingEdges";
  public static final String TMP_OPEN_TRIADS = "openTriads";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new EnumerateTrianglesJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption("text", "t", "output in textformat?", String.valueOf(Boolean.FALSE));

    Map<String, String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Class<? extends FileOutputFormat> outputFormat = Boolean.parseBoolean(parsedArgs.get("--text")) ?
        TextOutputFormat.class : SequenceFileOutputFormat.class;

    /* scatter the edges to each of the vertices and count degree */
    Job scatter = prepareJob(getInputPath(), getTempPath(TMP_AUGMENTED_EDGES), ScatterEdgesMapper.class,
        Vertex.class, Vertex.class, SumDegreesReducer.class, UndirectedEdge.class, VertexWithDegree.class);
    scatter.waitForCompletion(true);

    /* join augmented edges with partial degree information to to complete records */
    Job join = prepareJob(getTempPath(TMP_AUGMENTED_EDGES), getTempPath(TMP_EDGES_WITH_DEGREES), Mapper.class,
        UndirectedEdge.class, VertexWithDegree.class, JoinDegreesReducer.class, UndirectedEdgeWithDegrees.class,
        NullWritable.class);
    join.waitForCompletion(true);

    /* scatter the edges to lower degree vertex and build open triads */
    Job scatterToLower = prepareJob(getTempPath(TMP_EDGES_WITH_DEGREES), getTempPath(TMP_OPEN_TRIADS),
        ScatterEdgesToLowerDegreeVertexMapper.class, Vertex.class, Vertex.class, BuildOpenTriadsReducer.class,
        JoinableUndirectedEdge.class, VertexOrMarker.class);
    scatterToLower.waitForCompletion(true);

    /* necessary as long as we don't have access to an undeprecated MultipleInputs  */
    Job prepareInput = prepareJob(getTempPath(TMP_EDGES_WITH_DEGREES), getTempPath(TMP_CLOSING_EDGES),
        PrepareInputMapper.class, JoinableUndirectedEdge.class, VertexOrMarker.class, Reducer.class,
        JoinableUndirectedEdge.class, VertexOrMarker.class);
    prepareInput.setGroupingComparatorClass(JoinableUndirectedEdge.GroupingComparator.class);
    prepareInput.waitForCompletion(true);

    /* join opentriads and edges pairwise to get all triangles */
    Job joinTriads = prepareJob(getCombinedTempPath(TMP_OPEN_TRIADS, TMP_CLOSING_EDGES), getOutputPath(),
        SequenceFileInputFormat.class, Mapper.class, JoinableUndirectedEdge.class, VertexOrMarker.class,
        JoinTrianglesReducer.class, Triangle.class, NullWritable.class, outputFormat);
    joinTriads.setGroupingComparatorClass(JoinableUndirectedEdge.GroupingComparator.class);
    joinTriads.waitForCompletion(true);

    return 0;
  }

  /** Sends every edge to each vertex  */
  public static class ScatterEdgesMapper extends Mapper<UndirectedEdge,Object,Vertex,Vertex> {

    @Override
    protected void map(UndirectedEdge edge, Object value, Context ctx) throws IOException, InterruptedException {
      ctx.write(edge.firstVertex(), edge.secondVertex());
      ctx.write(edge.secondVertex(), edge.firstVertex());
    }
  }

  /** Sums up the count of edges for each vertex and augments all edges with a degree information for the key vertex */
  public static class SumDegreesReducer extends Reducer<Vertex, Vertex, UndirectedEdge, VertexWithDegree> {

    @Override
    protected void reduce(Vertex vertex, Iterable<Vertex> connectedVertices, Context ctx)
        throws IOException, InterruptedException {
      FastIDSet connectedVertexIds = new FastIDSet();
      for (Vertex connectedVertex : connectedVertices) {
        connectedVertexIds.add(connectedVertex.id());
      }

      int degree = connectedVertexIds.size();
      VertexWithDegree vertexWithDegree = new VertexWithDegree(vertex, degree);
      LongPrimitiveIterator connectedVertexIdsIterator = connectedVertexIds.iterator();
      while (connectedVertexIdsIterator.hasNext()) {
        Vertex connectedVertex = new Vertex(connectedVertexIdsIterator.nextLong());
        ctx.write(new UndirectedEdge(vertex, connectedVertex), vertexWithDegree);
      }
    }
  }

  /** Joins identical edges assuring degree augmentations for both nodes */
  public static class JoinDegreesReducer
      extends Reducer<UndirectedEdge,VertexWithDegree,UndirectedEdgeWithDegrees,NullWritable> {

    @Override
    protected void reduce(UndirectedEdge edge, Iterable<VertexWithDegree> verticesWithDegrees, Context ctx)
        throws IOException, InterruptedException {
      Iterator<VertexWithDegree> iterator = verticesWithDegrees.iterator();
      VertexWithDegree firstVertexWithDegree = iterator.next().clone();
      VertexWithDegree secondVertexWithDegree = iterator.next().clone();
      ctx.write(new UndirectedEdgeWithDegrees(firstVertexWithDegree, secondVertexWithDegree), NullWritable.get());
    }
  }


  /** Finds the lower degree vertex of an edge and emits key-value-pairs to bin under this lower degree vertex. */
  public static class ScatterEdgesToLowerDegreeVertexMapper extends
      Mapper<UndirectedEdgeWithDegrees,Writable,Vertex,Vertex> {

    @Override
    protected void map(UndirectedEdgeWithDegrees edge, Writable value, Context ctx)
        throws IOException, InterruptedException {
      VertexWithDegree first = edge.getFirstVertexWithDegree();
      VertexWithDegree second = edge.getSecondVertexWithDegree();

      if (first.degree() < second.degree()) {
        ctx.write(first.vertex(), second.vertex());
      } else {
        ctx.write(second.vertex(), first.vertex());
      }
    }
  }

  /**
   * Builds open triads from edges by pairwise joining the edges on the lower degree vertex which is the apex of the triad. Emits key-value pairs
   * where the value is the triad and the key is the two outside vertices.
   */
  public static class BuildOpenTriadsReducer extends Reducer<Vertex,Vertex,JoinableUndirectedEdge,VertexOrMarker> {

    @Override
    protected void reduce(Vertex vertex, Iterable<Vertex> vertices, Context ctx)
        throws IOException, InterruptedException {
      FastIDSet bufferedVertexIDs = new FastIDSet();
      for (Vertex firstVertexOfMissingEdge : vertices) {
        LongPrimitiveIterator bufferedVertexIdsIterator = bufferedVertexIDs.iterator();
        while (bufferedVertexIdsIterator.hasNext()) {
          Vertex secondVertexOfMissingEdge = new Vertex(bufferedVertexIdsIterator.nextLong());
          UndirectedEdge missingEdge = new UndirectedEdge(firstVertexOfMissingEdge, secondVertexOfMissingEdge);
          ctx.write(new JoinableUndirectedEdge(missingEdge, false), new VertexOrMarker(vertex));
        }
        bufferedVertexIDs.add(firstVertexOfMissingEdge.id());
      }
    }
  }

  public static class PrepareInputMapper
      extends Mapper<UndirectedEdgeWithDegrees,Writable,JoinableUndirectedEdge,VertexOrMarker> {

    @Override
    protected void map(UndirectedEdgeWithDegrees edgeWithDegrees, Writable value, Context ctx)
        throws IOException, InterruptedException {
      Vertex firstVertex = edgeWithDegrees.getFirstVertexWithDegree().vertex();
      Vertex secondVertex = edgeWithDegrees.getSecondVertexWithDegree().vertex();
      ctx.write(new JoinableUndirectedEdge(firstVertex, secondVertex, true), VertexOrMarker.MARKER);
    }
  }

  public static class JoinTrianglesReducer
      extends Reducer<JoinableUndirectedEdge,VertexOrMarker,Triangle,NullWritable> {

    @Override
    protected void reduce(JoinableUndirectedEdge joinableEdge, Iterable<VertexOrMarker> verticesAndMarker,
        Context ctx) throws IOException, InterruptedException {
      Iterator<VertexOrMarker> verticesAndMarkerIterator = verticesAndMarker.iterator();
      VertexOrMarker possibleMarker = verticesAndMarkerIterator.next();
      if (possibleMarker.isMarker()) {
        UndirectedEdge edge = joinableEdge.edge();
        while (verticesAndMarkerIterator.hasNext()) {
          Vertex connectingVertex = verticesAndMarkerIterator.next().vertex();
          ctx.write(new Triangle(connectingVertex, edge.firstVertex(), edge.secondVertex()), NullWritable.get());
        }
      }
    }
  }
}
