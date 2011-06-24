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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.reduce.IntSumReducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.graph.model.UndirectedEdge;
import org.apache.mahout.graph.model.Vertex;

import java.io.IOException;
import java.util.Map;

/**
 * <p>Distributed computation of the distribution of degrees of an undirected graph</p>
 *
 * <p>The input file needs to be a {@link org.apache.hadoop.io.SequenceFile} with {@link UndirectedEdge}s as keys and
 * any Writable as values, as it is already produced by {@link SimplifyGraphJob}</p>
 *
 * <p>This job outputs text files with a degree and the number of nodes having that degree per line.</p>
 *
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>-Dmapred.input.dir=(path): Directory containing one or more sequence files with edge data</li>
 * <li>-Dmapred.output.dir=(path): output path where the degree distribution data should be written</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public class DegreeDistributionJob extends AbstractJob {

  private static final IntWritable ONE = new IntWritable(1);

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new DegreeDistributionJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();

    Map<String, String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path tempDirPath = new Path(parsedArgs.get("--tempDir"));
    Path degreesPerVertexPath = new Path(tempDirPath, "degreesPerVertex");

    Job degreesPerVertex = prepareJob(getInputPath(), degreesPerVertexPath, SequenceFileInputFormat.class,
        DegreeOfVertexMapper.class, Vertex.class, IntWritable.class, IntSumReducer.class, Vertex.class,
        IntWritable.class, SequenceFileOutputFormat.class);
    degreesPerVertex.setCombinerClass(IntSumReducer.class);
    degreesPerVertex.waitForCompletion(true);

    Job degreeDistribution = prepareJob(degreesPerVertexPath, getOutputPath(), SequenceFileInputFormat.class,
        DegreesMapper.class, IntWritable.class, IntWritable.class, IntSumReducer.class, IntWritable.class,
        IntWritable.class, TextOutputFormat.class);
    degreeDistribution.setCombinerClass(IntSumReducer.class);
    degreeDistribution.waitForCompletion(true);

    return 0;
  }

  public static class DegreeOfVertexMapper extends Mapper<UndirectedEdge,Writable,Vertex,IntWritable> {
    @Override
    protected void map(UndirectedEdge edge, Writable value, Context ctx) throws IOException, InterruptedException {
      ctx.write(edge.getFirstVertex(), ONE);
      ctx.write(edge.getSecondVertex(), ONE);
    }
  }

  public static class DegreesMapper extends Mapper<Vertex,IntWritable,IntWritable,IntWritable> {
    @Override
    protected void map(Vertex vertex, IntWritable degree, Context ctx) throws IOException, InterruptedException {
      ctx.write(degree, ONE);
    }
  }
}
