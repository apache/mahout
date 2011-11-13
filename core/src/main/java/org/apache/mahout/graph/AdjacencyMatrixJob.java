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

package org.apache.mahout.graph;

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenIntIntHashMap;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * <p>Distributed computation of the adjacency matrix of a directed graph, see http://en.wikipedia.org/wiki/Adjacency_matrix
 *
 * <p>This job outputs {@link org.apache.hadoop.io.SequenceFile}s an {@link IntWritable} as key and a {@link VectorWritable}  as value</p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--output=(path): output path where the resulting matrix should be written</li>
 * <li>--vertices=(path): file containing a list of all vertices</li>
 * <li>--edges=(path): Directory containing edges of the graph</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public class AdjacencyMatrixJob extends AbstractJob {

  public static final String NUM_VERTICES = "numVertices.bin";
  public static final String ADJACENCY_MATRIX = "adjacencyMatrix";
  public static final String VERTEX_INDEX = "vertexIndex";

  static final String NUM_VERTICES_PARAM = AdjacencyMatrixJob.class.getName() + ".numVertices";
  static final String VERTEX_INDEX_PARAM = AdjacencyMatrixJob.class.getName() + ".vertexIndex";

  @Override
  public int run(String[] args) throws Exception {

    addOption("vertices", null, "a text file containing all vertices of the graph (one per line)", true);
    addOption("edges", null, "text files containing the edges of the graph (vertexA,vertextB per line)", true);
    addOutputOption();

    Map<String, String> parsedArgs = parseArguments(args);

    Path vertices = new Path(parsedArgs.get("--vertices"));
    Path edges = new Path(parsedArgs.get("--edges"));

    int numVertices = indexVertices(vertices, getOutputPath(VERTEX_INDEX));

    HadoopUtil.writeInt(numVertices, getOutputPath(NUM_VERTICES), getConf());

    Preconditions.checkArgument(numVertices > 0);

    Job createAdjacencyMatrix = prepareJob(edges, getOutputPath(ADJACENCY_MATRIX), TextInputFormat.class,
        VectorizeEdgesMapper.class, IntWritable.class, VectorWritable.class, VectorSumReducer.class,
        IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class);
    createAdjacencyMatrix.setCombinerClass(VectorSumReducer.class);
    Configuration createAdjacencyMatrixConf = createAdjacencyMatrix.getConfiguration();
    createAdjacencyMatrixConf.set(NUM_VERTICES_PARAM, String.valueOf(numVertices));
    createAdjacencyMatrixConf.set(VERTEX_INDEX_PARAM, getOutputPath(VERTEX_INDEX).toString());
    createAdjacencyMatrix.waitForCompletion(true);

    return 0;
  }

  //TODO do this in parallel?
  private int indexVertices(Path verticesPath, Path indexPath) throws IOException {
    FileSystem fs = FileSystem.get(verticesPath.toUri(), getConf());
    SequenceFile.Writer writer = null;
    int index = 0;

    try {
      writer = SequenceFile.createWriter(fs, getConf(), indexPath, IntWritable.class, IntWritable.class);

      for (FileStatus fileStatus : fs.listStatus(verticesPath)) {
        InputStream in = null;
        try {
          in = HadoopUtil.openStream(fileStatus.getPath(), getConf());
          for (String line : new FileLineIterable(in)) {
            writer.append(new IntWritable(index++), new IntWritable(Integer.parseInt(line)));
          }
        } finally {
          Closeables.closeQuietly(in);
        }
      }
    } finally {
      Closeables.closeQuietly(writer);
    }

    return index;
  }

  static class VectorizeEdgesMapper extends Mapper<LongWritable,Text,IntWritable,VectorWritable> {

    private int numVertices;
    private OpenIntIntHashMap vertexIDsToIndex;

    private final IntWritable row = new IntWritable();

    private static final Pattern SEPARATOR = Pattern.compile("[\t,]");

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      Configuration conf = ctx.getConfiguration();
      numVertices = Integer.parseInt(conf.get(NUM_VERTICES_PARAM));
      Path vertexIndexPath = new Path(conf.get(VERTEX_INDEX_PARAM));
      vertexIDsToIndex = new OpenIntIntHashMap(numVertices);
      for (Pair<IntWritable,IntWritable> indexAndVertexID :
          new SequenceFileIterable<IntWritable,IntWritable>(vertexIndexPath, true, conf)) {
        vertexIDsToIndex.put(indexAndVertexID.getSecond().get(), indexAndVertexID.getFirst().get());
      }
    }

    @Override
    protected void map(LongWritable offset, Text line, Mapper.Context ctx)
        throws IOException, InterruptedException {

      String[] tokens = SEPARATOR.split(line.toString());
      int rowIndex = vertexIDsToIndex.get(Integer.parseInt(tokens[0]));
      int columnIndex = vertexIDsToIndex.get(Integer.parseInt(tokens[1]));
      RandomAccessSparseVector partialTransitionMatrixRow = new RandomAccessSparseVector(numVertices, 1);

      row.set(rowIndex);
      partialTransitionMatrixRow.setQuick(columnIndex, 1);

      ctx.write(row, new VectorWritable(partialTransitionMatrixRow));
    }
  }

}