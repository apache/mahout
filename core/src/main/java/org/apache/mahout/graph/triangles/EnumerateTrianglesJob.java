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

package org.apache.mahout.graph.triangles;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.graph.model.Triangle;
import org.apache.mahout.graph.model.UndirectedEdge;
import org.apache.mahout.graph.model.UndirectedEdgeWithDegrees;
import org.apache.mahout.graph.model.Vertex;
import org.apache.mahout.graph.model.VertexWithDegree;

/** Enumerates all triangles of an undirected graph. */
public class EnumerateTrianglesJob extends AbstractJob {

  public static final String TMP_CLOSING_EDGES = "closingEdges";
  public static final String TMP_OPEN_TRIADS = "openTriads";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new EnumerateTrianglesJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();

    Map<String, String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    // scatter the edges to lower degree vertex and build open triads
    Job scatter = prepareJob(getInputPath(), getTempPath(TMP_OPEN_TRIADS), SequenceFileInputFormat.class,
        ScatterEdgesToLowerDegreeVertexMapper.class, Vertex.class, Vertex.class,
        BuildOpenTriadsReducer.class, JoinableUndirectedEdge.class, VertexOrMarker.class,
        SequenceFileOutputFormat.class);
    scatter.waitForCompletion(true);

    // necessary as long as we don't have access to an undeprecated MultipleInputs
    Job prepareInput = prepareJob(getInputPath(), getTempPath(TMP_CLOSING_EDGES), SequenceFileInputFormat.class,
        PrepareInputMapper.class, JoinableUndirectedEdge.class, VertexOrMarker.class, Reducer.class,
        JoinableUndirectedEdge.class, VertexOrMarker.class, SequenceFileOutputFormat.class);
    prepareInput.setGroupingComparatorClass(JoinableUndirectedEdge.GroupingComparator.class);
    prepareInput.waitForCompletion(true);

    //join opentriads and edges pairwise to get all triangles
    Job joinTriads = prepareJob(getCombinedTempPath(TMP_OPEN_TRIADS, TMP_CLOSING_EDGES), getOutputPath(),
        SequenceFileInputFormat.class, Mapper.class, JoinableUndirectedEdge.class, VertexOrMarker.class,
        JoinTrianglesReducer.class, Triangle.class, NullWritable.class, SequenceFileOutputFormat.class);
    joinTriads.setGroupingComparatorClass(JoinableUndirectedEdge.GroupingComparator.class);
    joinTriads.waitForCompletion(true);

    return 0;
  }

  /** Finds the lower degree vertex of an edge and emits key-value-pairs to bin under this lower degree vertex. */
  public static class ScatterEdgesToLowerDegreeVertexMapper extends
      Mapper<UndirectedEdgeWithDegrees,Object,Vertex,Vertex> {
    @Override
    protected void map(UndirectedEdgeWithDegrees edge, Object value, Context ctx)
        throws IOException, InterruptedException {
      VertexWithDegree first = edge.getFirstVertexWithDegree();
      VertexWithDegree second = edge.getSecondVertexWithDegree();

      if (first.getDegree() < second.getDegree()) {
        ctx.write(first.getVertex(), second.getVertex());
      } else {
        ctx.write(second.getVertex(), first.getVertex());
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
          System.out.println(new JoinableUndirectedEdge(missingEdge, false) + " " + new VertexOrMarker(vertex));
          ctx.write(new JoinableUndirectedEdge(missingEdge, false), new VertexOrMarker(vertex));
        }
        bufferedVertexIDs.add(firstVertexOfMissingEdge.getId());
      }
    }
  }

  public static class PrepareInputMapper
      extends Mapper<UndirectedEdgeWithDegrees,Object,JoinableUndirectedEdge,VertexOrMarker> {
    @Override
    protected void map(UndirectedEdgeWithDegrees edgeWithDegrees, Object value, Context ctx)
        throws IOException, InterruptedException {
      Vertex firstVertex = edgeWithDegrees.getFirstVertexWithDegree().getVertex();
      Vertex secondVertex = edgeWithDegrees.getSecondVertexWithDegree().getVertex();
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
        UndirectedEdge edge = joinableEdge.getEdge();
        while (verticesAndMarkerIterator.hasNext()) {
          Vertex connectingVertex = verticesAndMarkerIterator.next().getVertex();
          ctx.write(new Triangle(connectingVertex, edge.getFirstVertex(), edge.getSecondVertex()), NullWritable.get());
        }
      }
    }
  }
}
