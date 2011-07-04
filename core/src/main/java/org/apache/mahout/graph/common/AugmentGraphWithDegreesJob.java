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

import org.apache.hadoop.fs.Path;
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
import org.apache.mahout.graph.model.UndirectedEdge;
import org.apache.mahout.graph.model.UndirectedEdgeWithDegrees;
import org.apache.mahout.graph.model.Vertex;
import org.apache.mahout.graph.model.VertexWithDegree;

/**
 * Augments a graph with degree information for each vertex which is the number
 * of {@link org.apache.mahout.graph.model.UndirectedEdge}s that point to or from this very vertex.
 */
public class AugmentGraphWithDegreesJob extends AbstractJob {

  public static final String TMP_AUGMENTED_EDGES = "augmented-edges";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new AugmentGraphWithDegreesJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();

    Map<String, String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();

    // scatter the edges to each of the vertices and count degree
    Job scatter = prepareJob(inputPath, getTempPath(TMP_AUGMENTED_EDGES), SequenceFileInputFormat.class,
        ScatterEdgesMapper.class, Vertex.class, Vertex.class, SumDegreesReducer.class, UndirectedEdge.class,
        VertexWithDegree.class, SequenceFileOutputFormat.class);
    scatter.waitForCompletion(true);

    // join augmented edges with partial degree information to to complete records
    Job join = prepareJob(getTempPath(TMP_AUGMENTED_EDGES), outputPath, SequenceFileInputFormat.class,
        Mapper.class, UndirectedEdge.class, VertexWithDegree.class, JoinDegreesReducer.class,
        UndirectedEdgeWithDegrees.class, NullWritable.class, SequenceFileOutputFormat.class);
    join.waitForCompletion(true);

    return 0;
  }

  /** Sends every edge to each vertex  */
  public static class ScatterEdgesMapper extends Mapper<UndirectedEdge,Object,Vertex,Vertex> {
    @Override
    protected void map(UndirectedEdge edge, Object value, Context ctx) throws IOException, InterruptedException {
      ctx.write(edge.getFirstVertex(), edge.getSecondVertex());
      ctx.write(edge.getSecondVertex(), edge.getFirstVertex());
    }
  }

  /** Sums up the count of edges for each vertex and augments all edges with a degree information for the key vertex */
  public static class SumDegreesReducer extends Reducer<Vertex, Vertex, UndirectedEdge, VertexWithDegree> {
    @Override
    protected void reduce(Vertex vertex, Iterable<Vertex> connectedVertices, Context ctx)
        throws IOException, InterruptedException {
      FastIDSet connectedVertexIds = new FastIDSet();
      for (Vertex connectedVertex : connectedVertices) {
        connectedVertexIds.add(connectedVertex.getId());
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
}
