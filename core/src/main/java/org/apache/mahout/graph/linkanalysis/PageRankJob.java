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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Map;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.reduce.IntSumReducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.graph.model.Edge;
import org.apache.mahout.graph.model.Vertex;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

import com.google.common.io.Closeables;
import org.apache.mahout.math.map.OpenLongIntHashMap;

/**
 * <p>Distributed computation of the PageRank a directed graph</p>
 *
 * <p>The input files need to be a {@link org.apache.hadoop.io.SequenceFile} with {@link Edge}s as keys and
 * any Writable as values and another {@link org.apache.hadoop.io.SequenceFile} with {@link IntWritable}s as keys and {@link Vertex} as
 * values, as produced by {@link org.apache.mahout.graph.common.GraphUtils#indexVertices(Configuration, Path, Path)}</p>
 *
 * <p>This job outputs text files with a vertex id and its pagerank per line.</p>
  *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>-Dmapred.output.dir=(path): output path where the degree distribution data should be written</li>
 * <li>--vertexIndex=(path): Directory containing vertex index as created by GraphUtils.indexVertices()</li>
 * <li>--edges=(path): Directory containing edges of the graph</li>
 * <li>--numVertices=(Integer): number of vertices in the graph</li>
 * <li>--numIterations=(Integer): number of numIterations, default: 5</li>
 * <li>--teleportationProbability=(Double): probability to teleport to a random vertex, default: 0.8</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public class PageRankJob extends AbstractJob {

  static final String TMP_INDEXED_DEGREES = "indexedDegrees";
  static final String TMP_TRANSITION_MATRIX = "transitionMatrix";
  static final String TMP_PAGERANK_VECTOR = "pageRankVector";

  static final String NUM_VERTICES_PARAM = PageRankJob.class.getName() + ".numVertices";
  static final String VERTEX_INDEX_PARAM = PageRankJob.class.getName() + ".vertexIndex";
  static final String INDEXED_DEGREES_PARAM = PageRankJob.class.getName() + ".indexedDegrees";
  static final String TELEPORTATION_PROBABILITY_PARAM = PageRankJob.class.getName() + ".teleportationProbability";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new PageRankJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addOutputOption();
    addOption("vertexIndex", "vi", "vertex index as created by GraphUtils.indexVertices()", true);
    addOption("edges", "e", "edges of the graph", true);
    addOption("numVertices", "nn", "number of vertices in the graph", true);
    addOption("numIterations", "it", "number of numIterations", String.valueOf(5));
    addOption("teleportationProbability", "tp", "probability to teleport to a random vertex", String.valueOf(0.8));

    Map<String, String> parsedArgs = parseArguments(args);

    Path vertexIndex = new Path(parsedArgs.get("--vertexIndex"));
    Path edges = new Path(parsedArgs.get("--edges"));

    int numVertices = Integer.parseInt(parsedArgs.get("--numVertices"));
    int numIterations = Integer.parseInt(parsedArgs.get("--numIterations"));
    double teleportationProbability = Double.parseDouble(parsedArgs.get("--teleportationProbability"));

    Preconditions.checkArgument(numVertices > 0);
    Preconditions.checkArgument(numIterations > 0);
    Preconditions.checkArgument(teleportationProbability > 0.0 && teleportationProbability < 1.0);

    Job indexedDegrees = prepareJob(edges, getTempPath(TMP_INDEXED_DEGREES), SequenceFileInputFormat.class,
        IndexAndCountDegreeMapper.class, IntWritable.class, IntWritable.class, IntSumReducer.class, IntWritable.class,
        IntWritable.class, SequenceFileOutputFormat.class);
    indexedDegrees.getConfiguration().set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    indexedDegrees.getConfiguration().set(VERTEX_INDEX_PARAM, vertexIndex.toString());
    indexedDegrees.setCombinerClass(IntSumReducer.class);
    indexedDegrees.waitForCompletion(true);

    Job createTransitionMatrix = prepareJob(edges, getTempPath(TMP_TRANSITION_MATRIX),
        SequenceFileInputFormat.class, RevertEdgesMapper.class, IntWritable.class, IntWritable.class,
        CreateTransitionMatrixReducer.class, IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
    createTransitionMatrix.getConfiguration().set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    createTransitionMatrix.getConfiguration().set(VERTEX_INDEX_PARAM, vertexIndex.toString());
    createTransitionMatrix.getConfiguration().set(INDEXED_DEGREES_PARAM,
        getTempPath(TMP_INDEXED_DEGREES).toString());
    createTransitionMatrix.getConfiguration().set(TELEPORTATION_PROBABILITY_PARAM,
        String.valueOf(teleportationProbability));
    createTransitionMatrix.waitForCompletion(true);

    DistributedRowMatrix matrix = new DistributedRowMatrix(getTempPath(TMP_TRANSITION_MATRIX), getTempPath(),
        numVertices, numVertices);
    matrix.setConf(getConf());

    Vector pageRank = new DenseVector(numVertices).assign(1.0 / numVertices);
    Vector damplingVector = new DenseVector(numVertices).assign((1.0 - teleportationProbability) / numVertices);

    while (numIterations-- > 0) {
      pageRank = matrix.times(pageRank).plus(damplingVector);
    }

    FileSystem fs = FileSystem.get(getTempPath(TMP_PAGERANK_VECTOR).toUri(), getConf());
    DataOutputStream stream = fs.create(getTempPath(TMP_PAGERANK_VECTOR), true);
    try {
      VectorWritable.writeVector(stream, pageRank);
    } finally {
      Closeables.closeQuietly(stream);
    }

    Job vertexWithPageRank = prepareJob(vertexIndex, getOutputPath(), SequenceFileInputFormat.class,
        VertexWithPageRankMapper.class, LongWritable.class, DoubleWritable.class, Reducer.class, LongWritable.class,
        DoubleWritable.class, TextOutputFormat.class);
    vertexWithPageRank.getConfiguration().set(VertexWithPageRankMapper.PAGERANK_PATH_PARAM,
        getTempPath(TMP_PAGERANK_VECTOR).toString());
    vertexWithPageRank.waitForCompletion(true);

    return 1;
  }

  public static class IndexAndCountDegreeMapper extends Mapper<Edge,Writable,IntWritable,IntWritable> {

    private OpenLongIntHashMap vertexIDsToIndex;

    private static final IntWritable ONE = new IntWritable(1);

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();
      int numVertices = Integer.parseInt(conf.get(NUM_VERTICES_PARAM));
      Path vertexIndexPath = new Path(conf.get(VERTEX_INDEX_PARAM));
      vertexIDsToIndex = new OpenLongIntHashMap(numVertices);
      for (Pair<IntWritable,Vertex> indexAndVertexID :
          new SequenceFileIterable<IntWritable,Vertex>(vertexIndexPath, true, conf)) {
        vertexIDsToIndex.put(indexAndVertexID.getSecond().getId(), indexAndVertexID.getFirst().get());
      }
    }

    @Override
    protected void map(Edge edge, Writable value, Context ctx) throws IOException, InterruptedException {
      int startIndex = vertexIDsToIndex.get(edge.startVertex().getId());
      ctx.write(new IntWritable(startIndex), ONE);
    }
  }

  public static class RevertEdgesMapper extends Mapper<Edge,Writable,IntWritable,IntWritable> {

    private OpenLongIntHashMap vertexIDsToIndex;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();
      int numVertices = Integer.parseInt(conf.get(NUM_VERTICES_PARAM));
      Path vertexIndexPath = new Path(conf.get(VERTEX_INDEX_PARAM));
      vertexIDsToIndex = new OpenLongIntHashMap(numVertices);
      for (Pair<IntWritable,Vertex> indexAndVertexID :
          new SequenceFileIterable<IntWritable,Vertex>(vertexIndexPath, true, conf)) {
        vertexIDsToIndex.put(indexAndVertexID.getSecond().getId(), indexAndVertexID.getFirst().get());
      }
    }

    @Override
    protected void map(Edge edge, Writable value, Context ctx) throws IOException, InterruptedException {
      int startIndex = vertexIDsToIndex.get(edge.startVertex().getId());
      int endIndex = vertexIDsToIndex.get(edge.endVertex().getId());
      ctx.write(new IntWritable(endIndex), new IntWritable(startIndex));
    }
  }

  public static class CreateTransitionMatrixReducer
      extends Reducer<IntWritable, IntWritable, IntWritable, VectorWritable> {

    private int numVertices;
    private double teleportationProbability;
    private Vector weights;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();
      Path indexedDegreesPath = new Path(ctx.getConfiguration().get(INDEXED_DEGREES_PARAM));
      numVertices = Integer.parseInt(conf.get(NUM_VERTICES_PARAM));
      teleportationProbability = Double.parseDouble(conf.get(TELEPORTATION_PROBABILITY_PARAM));
      Preconditions.checkArgument(numVertices > 0);
      Preconditions.checkArgument(teleportationProbability > 0 && teleportationProbability < 1);
      weights = new DenseVector(numVertices);

      for (Pair<IntWritable, IntWritable> indexAndDegree :
          new SequenceFileDirIterable<IntWritable, IntWritable>(indexedDegreesPath, PathType.LIST,
          PathFilters.partFilter(), ctx.getConfiguration())) {
        weights.set(indexAndDegree.getFirst().get(), 1.0 / indexAndDegree.getSecond().get());
      }
    }

    @Override
    protected void reduce(IntWritable vertexIndex, Iterable<IntWritable> incidentVertexIndexes, Context ctx)
        throws IOException, InterruptedException {
      Vector vector = new RandomAccessSparseVector(numVertices);
      for (IntWritable incidentVertexIndex : incidentVertexIndexes) {
        double weight = weights.get(incidentVertexIndex.get()) * teleportationProbability;
        //System.out.println(vertexIndex.get() + "," + incidentVertexIndex.get() + ": " + weight);
        vector.set(incidentVertexIndex.get(), weight);
      }
      ctx.write(vertexIndex, new VectorWritable(vector));
    }
  }

  public static class VertexWithPageRankMapper extends Mapper<IntWritable,Vertex,LongWritable,DoubleWritable> {

    static final String PAGERANK_PATH_PARAM = VertexWithPageRankMapper.class.getName() + ".pageRankPath";

    private Vector pageRanks;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Path pageRankPath = new Path(ctx.getConfiguration().get(PAGERANK_PATH_PARAM));
      DataInputStream in = FileSystem.get(pageRankPath.toUri(), ctx.getConfiguration()).open(pageRankPath);
      try {
        pageRanks = VectorWritable.readVector(in);
      } finally {
        Closeables.closeQuietly(in);
      }
    }

    @Override
    protected void map(IntWritable index, Vertex vertex, Context ctx) throws IOException, InterruptedException {
      ctx.write(new LongWritable(vertex.getId()), new DoubleWritable(pageRanks.get(index.get())));
    }
  }

}
