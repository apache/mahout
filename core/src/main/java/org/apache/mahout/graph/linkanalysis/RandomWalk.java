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

package org.apache.mahout.graph.linkanalysis;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.graph.model.Vertex;
import org.apache.mahout.graph.preprocessing.AdjacencyMatrixJob;
import org.apache.mahout.graph.preprocessing.GraphUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Map;

public abstract class RandomWalk extends AbstractJob {

  static final String ADJACENCY_MATRIX = "adjacencyMatrix";
  static final String RANK_VECTOR = "rankVector";

  protected abstract Vector createDampingVector(int numVertices, double stayingProbability);

  protected void addSpecificOptions() {}
  protected void evaluateSpecificOptions(Map<String, String> parsedArgs) {}

  @Override
  public final int run(String[] args) throws Exception {
    addOutputOption();
    addOption("vertexIndex", "vi", "vertex index as created by GraphUtils.indexVertices()", true);
    addOption("edges", "e", "edges of the graph", true);
    addOption("numVertices", "nv", "number of vertices in the graph", true);
    addOption("numIterations", "it", "number of numIterations", String.valueOf(5));
    addOption("stayingProbability", "tp", "probability not to teleport to a random vertex", String.valueOf(0.8));

    addSpecificOptions();

    Map<String, String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    evaluateSpecificOptions(parsedArgs);

    Path vertexIndex = new Path(parsedArgs.get("--vertexIndex"));
    Path edges = new Path(parsedArgs.get("--edges"));

    int numVertices = Integer.parseInt(parsedArgs.get("--numVertices"));
    int numIterations = Integer.parseInt(parsedArgs.get("--numIterations"));
    double stayingProbability = Double.parseDouble(parsedArgs.get("--stayingProbability"));

    Preconditions.checkArgument(numVertices > 0);
    Preconditions.checkArgument(numIterations > 0);
    Preconditions.checkArgument(stayingProbability > 0.0 && stayingProbability <= 1.0);

    /* create the substochastified adjacency matrix */
    ToolRunner.run(getConf(), new AdjacencyMatrixJob(), new String[] { "--vertexIndex", vertexIndex.toString(),
        "--edges", edges.toString(), "--output", getTempPath(ADJACENCY_MATRIX).toString(),
        "--numVertices", String.valueOf(numVertices), "--stayingProbability", String.valueOf(stayingProbability),
        "--substochastify", String.valueOf(true), "--tempDir", getTempPath().toString() });

    DistributedRowMatrix adjacencyMatrix = new DistributedRowMatrix(getTempPath(ADJACENCY_MATRIX), getTempPath(),
        numVertices, numVertices);
    adjacencyMatrix.setConf(getConf());

    Vector ranking = new DenseVector(numVertices).assign(1.0 / numVertices);
    Vector dampingVector = createDampingVector(numVertices, stayingProbability);

    /* power method: iterative adjacency-matrix times ranking-vector multiplication */
    while (numIterations-- > 0) {
      ranking = adjacencyMatrix.times(ranking).plus(dampingVector);
    }

    GraphUtils.persistVector(getConf(), getTempPath(RANK_VECTOR), ranking);

    Job vertexWithPageRank = prepareJob(vertexIndex, getOutputPath(), SequenceFileInputFormat.class,
        RankPerVertexMapper.class, LongWritable.class, DoubleWritable.class, TextOutputFormat.class);
    vertexWithPageRank.getConfiguration().set(RankPerVertexMapper.RANK_PATH_PARAM,
        getTempPath(RANK_VECTOR).toString());
    vertexWithPageRank.waitForCompletion(true);

    return 1;
  }

  public static class RankPerVertexMapper extends Mapper<IntWritable,Vertex,LongWritable,DoubleWritable> {

    static final String RANK_PATH_PARAM = RankPerVertexMapper.class.getName() + ".pageRankPath";

    private Vector ranks;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Path pageRankPath = new Path(ctx.getConfiguration().get(RANK_PATH_PARAM));
      DataInputStream in = FileSystem.get(pageRankPath.toUri(), ctx.getConfiguration()).open(pageRankPath);
      try {
        ranks = VectorWritable.readVector(in);
      } finally {
        Closeables.closeQuietly(in);
      }
    }

    protected void map(IntWritable index, Vertex vertex, Mapper.Context ctx) throws IOException, InterruptedException {
      ctx.write(new LongWritable(vertex.id()), new DoubleWritable(ranks.get(index.get())));
    }
  }

}
