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

package org.apache.mahout.graph.preprocessing;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.graph.model.Edge;
import org.apache.mahout.graph.model.Vertex;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.map.OpenLongIntHashMap;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

/**
 * <p>Distributed computation of the adjacency matrix of a directed graph, see http://en.wikipedia.org/wiki/Adjacency_matrix, with the
 * option for normalizing it row-wise and multiplying it with teleportation probabilities as necessary for
 * {@link org.apache.mahout.graph.linkanalysis.PageRankJob} or {@link org.apache.mahout.graph.linkanalysis.RandomWalkWithRestartJob}</p>
 *
 * <p>This job outputs {@link org.apache.hadoop.io.SequenceFile}s an {@link IntWritable} as key and a {@link VectorWritable}  as value</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--output=(path): output path where the resulting matrix should be written</li>
 * <li>--vertexIndex=(path): Directory containing vertex index as created by GraphUtils.indexVertices()</li>
 * <li>--edges=(path): Directory containing edges of the graph</li>
 * <li>--numVertices=(Integer): number of vertices in the graph</li>
 * <li>--stayingProbability=(Double): probability not to teleport to a random vertex, default: 1</li>
 * <li>--normalize=(boolean): normalize the rows of the resulting matrix, default: false</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public class AdjacencyMatrixJob extends AbstractJob {

  static final String NUM_VERTICES_PARAM = AdjacencyMatrixJob.class.getName() + ".numVertices";
  static final String STAYING_PROBABILITY_PARAM = AdjacencyMatrixJob.class.getName() + ".stayingProbability";
  static final String VERTEX_INDEX_PARAM = AdjacencyMatrixJob.class.getName() + ".vertexIndex";
  static final String STOCHASTIFY_PARAM = AdjacencyMatrixJob.class.getName() + ".normalize";

  private static final String TRANSPOSED_ADJACENCY_MATRIX = "transposedAdjacencyMatrix";

  @Override
  public int run(String[] args) throws Exception {

    addOption("vertexIndex", "vi", "vertex index as created by GraphUtils.indexVertices()", true);
    addOption("edges", "e", "edges of the graph", true);
    addOption("numVertices", "nv", "number of vertices in the graph", true);
    addOption("stayingProbability", "sp", "probability not to teleport to another vertex", String.valueOf(1));
    addOption("substochastify", "st", "substochastify the adjacency matrix?", String.valueOf(false));
    addOutputOption();

    Map<String, String> parsedArgs = parseArguments(args);

    Path vertexIndex = new Path(parsedArgs.get("--vertexIndex"));
    Path edges = new Path(parsedArgs.get("--edges"));
    int numVertices = Integer.parseInt(parsedArgs.get("--numVertices"));
    double stayingProbability = Double.parseDouble(parsedArgs.get("--stayingProbability"));
    boolean stochastify = Boolean.parseBoolean(parsedArgs.get("--substochastify"));

    Preconditions.checkArgument(numVertices > 0);
    Preconditions.checkArgument(stayingProbability > 0 && stayingProbability <= 1);

    Job createTransposedAdjacencyMatrix = prepareJob(edges, getTempPath(TRANSPOSED_ADJACENCY_MATRIX),
        VectorizeEdgesMapper.class, IntWritable.class, VectorWritable.class, SubstochastifyingVectorSumReducer.class,
        IntWritable.class, VectorWritable.class);
    createTransposedAdjacencyMatrix.setCombinerClass(VectorSumReducer.class);
    Configuration createAdjacencyMatrixConf = createTransposedAdjacencyMatrix.getConfiguration();
    createAdjacencyMatrixConf.set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    createAdjacencyMatrixConf.set(VERTEX_INDEX_PARAM, vertexIndex.toString());
    createAdjacencyMatrixConf.set(STAYING_PROBABILITY_PARAM, String.valueOf(stayingProbability));
    createAdjacencyMatrixConf.set(STOCHASTIFY_PARAM, String.valueOf(stochastify));
    createTransposedAdjacencyMatrix.waitForCompletion(true);

    Job transposeTransposedAdjacencyMatrix = prepareJob(getTempPath(TRANSPOSED_ADJACENCY_MATRIX), getOutputPath(),
        TransposeMapper.class, IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class,
        VectorWritable.class);
    transposeTransposedAdjacencyMatrix.setCombinerClass(MergeVectorsCombiner.class);
    transposeTransposedAdjacencyMatrix.getConfiguration().set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    transposeTransposedAdjacencyMatrix.waitForCompletion(true);

    return 0;
  }

  static class VectorizeEdgesMapper extends Mapper<Edge,Writable,IntWritable,VectorWritable> {

    private int numVertices;
    private OpenLongIntHashMap vertexIDsToIndex;

    private final IntWritable row = new IntWritable();

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();
      numVertices = Integer.parseInt(conf.get(NUM_VERTICES_PARAM));
      Path vertexIndexPath = new Path(conf.get(VERTEX_INDEX_PARAM));
      vertexIDsToIndex = new OpenLongIntHashMap(numVertices);
      for (Pair<IntWritable,Vertex> indexAndVertexID :
          new SequenceFileIterable<IntWritable,Vertex>(vertexIndexPath, true, conf)) {
        vertexIDsToIndex.put(indexAndVertexID.getSecond().id(), indexAndVertexID.getFirst().get());
      }
    }

    @Override
    protected void map(Edge edge, Writable value, Mapper.Context ctx) throws IOException, InterruptedException {
      int rowIndex = vertexIDsToIndex.get(edge.startVertex().id());
      int columnIndex = vertexIDsToIndex.get(edge.endVertex().id());
      RandomAccessSparseVector partialTransitionMatrixRow = new RandomAccessSparseVector(numVertices, 1);

      row.set(rowIndex);
      partialTransitionMatrixRow.setQuick(columnIndex, 1);

      ctx.write(row, new VectorWritable(partialTransitionMatrixRow));
    }
  }

  static class SubstochastifyingVectorSumReducer
      extends Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

    private double stayingProbability;
    private boolean normalize;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      normalize = ctx.getConfiguration().getBoolean(STOCHASTIFY_PARAM, false);
      stayingProbability = Double.parseDouble(ctx.getConfiguration().get(STAYING_PROBABILITY_PARAM));
    }

    @Override
    protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context context)
        throws IOException, InterruptedException {
      Vector vector = null;
      for (VectorWritable v : values) {
        if (vector == null) {
          vector = v.get();
        } else {
          vector.assign(v.get(), Functions.PLUS);
        }
      }
      if (normalize) {
        vector = vector.normalize(1);
      }
      if (stayingProbability != 1.0) {
        vector.assign(Functions.MULT, stayingProbability);
      }

      context.write(key, new VectorWritable(vector));
    }
  }

  static class TransposeMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private int numVertices;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      numVertices = Integer.parseInt(ctx.getConfiguration().get(NUM_VERTICES_PARAM));
    }

    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
      int row = r.get();
      Iterator<Vector.Element> it = v.get().iterateNonZero();
      while (it.hasNext()) {
        Vector.Element e = it.next();
        RandomAccessSparseVector tmp = new RandomAccessSparseVector(numVertices, 1);
        tmp.setQuick(row, e.get());
        r.set(e.index());
        ctx.write(r, new VectorWritable(tmp));
      }
    }
  }

  public static class MergeVectorsCombiner
      extends Reducer<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {

    @Override
    public void reduce(WritableComparable<?> key, Iterable<VectorWritable> vectors, Context ctx)
        throws IOException, InterruptedException {
      ctx.write(key, VectorWritable.merge(vectors.iterator()));
    }
  }

  public static class MergeVectorsReducer extends
      Reducer<WritableComparable<?>,VectorWritable,WritableComparable<?>,VectorWritable> {

    @Override
    public void reduce(WritableComparable<?> key, Iterable<VectorWritable> vectors, Context ctx)
        throws IOException, InterruptedException {
      Vector merged = VectorWritable.merge(vectors.iterator()).get();
      ctx.write(key, new VectorWritable(new SequentialAccessSparseVector(merged)));
    }
  }

}