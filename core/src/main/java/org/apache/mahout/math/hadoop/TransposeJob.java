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

package org.apache.mahout.math.hadoop;

import org.apache.commons.cli2.Option;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

/**
 * TODO: rewrite to use helpful combiner.
 */
public class TransposeJob extends AbstractJob {
  public static final String NUM_ROWS_KEY = "SparseRowMatrix.numRows";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new TransposeJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    Option numRowsOpt = buildOption("numRows",
                                    "nr",
                                    "Number of rows of the input matrix");
    Option numColsOpt = buildOption("numCols",
                                    "nc",
                                    "Number of columns of the input matrix");
    Map<String,String> parsedArgs = parseArguments(strings, numRowsOpt, numColsOpt);

    Configuration originalConf = getConf();
    String inputPathString = originalConf.get("mapred.input.dir");
    String outputTmpPathString = parsedArgs.get("--tempDir");
    int numRows = Integer.parseInt(parsedArgs.get("--numRows"));
    int numCols = Integer.parseInt(parsedArgs.get("--numCols"));

    DistributedRowMatrix matrix = new DistributedRowMatrix(inputPathString, outputTmpPathString, numRows, numCols);
    matrix.configure(new JobConf(getConf()));
    matrix.transpose();

    return 0;
  }

  public static JobConf buildTransposeJobConf(Path matrixInputPath,
                                              Path matrixOutputPath,
                                              int numInputRows) throws IOException {
    JobConf conf = new JobConf(TransposeJob.class);
    conf.setJobName("TransposeJob: " + matrixInputPath + " transpose -> " + matrixOutputPath);
    FileSystem fs = FileSystem.get(conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);
    conf.setInt(NUM_ROWS_KEY, numInputRows);

    FileInputFormat.addInputPath(conf, matrixInputPath);
    conf.setInputFormat(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(conf, matrixOutputPath);
    conf.setMapperClass(TransposeMapper.class);
    conf.setReducerClass(TransposeReducer.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(DistributedRowMatrix.MatrixEntryWritable.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    return conf;
  }

  public static class TransposeMapper extends MapReduceBase
      implements Mapper<IntWritable,VectorWritable,IntWritable,DistributedRowMatrix.MatrixEntryWritable> {

    @Override
    public void map(IntWritable r,
                    VectorWritable v,
                    OutputCollector<IntWritable, DistributedRowMatrix.MatrixEntryWritable> out,
                    Reporter reporter) throws IOException {
      DistributedRowMatrix.MatrixEntryWritable entry = new DistributedRowMatrix.MatrixEntryWritable();
      Iterator<Vector.Element> it = v.get().iterateNonZero();
      int row = r.get();
      entry.setCol(row);
      entry.setRow(-1);  // output "row" is captured in the key
      while (it.hasNext()) {
        Vector.Element e = it.next();
        r.set(e.index());
        entry.setVal(e.get());
        out.collect(r, entry);
      }
    }
  }

  public static class TransposeReducer extends MapReduceBase
      implements Reducer<IntWritable,DistributedRowMatrix.MatrixEntryWritable,IntWritable,VectorWritable> {

    //private JobConf conf;
    private int newNumCols;

    @Override
    public void configure(JobConf conf) {
      //this.conf = conf;
      newNumCols = conf.getInt(NUM_ROWS_KEY, Integer.MAX_VALUE);
    }

    @Override
    public void reduce(IntWritable outRow,
                       Iterator<DistributedRowMatrix.MatrixEntryWritable> it,
                       OutputCollector<IntWritable, VectorWritable> out,
                       Reporter reporter) throws IOException {
      RandomAccessSparseVector tmp = new RandomAccessSparseVector(newNumCols, 100);
      while (it.hasNext()) {
        DistributedRowMatrix.MatrixEntryWritable e = it.next();
        tmp.setQuick(e.getCol(), e.getVal());
      }
      SequentialAccessSparseVector outVector = new SequentialAccessSparseVector(tmp);
      out.collect(outRow, new VectorWritable(outVector));
    }
  }
}
