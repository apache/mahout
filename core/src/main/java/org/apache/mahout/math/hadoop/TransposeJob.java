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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;
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
import java.util.List;
import java.util.Map;

/**
 * Transpose a matrix
 */
public class TransposeJob extends AbstractJob {

  public static final String NUM_ROWS_KEY = "SparseRowMatrix.numRows";

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new TransposeJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    addInputOption();
    addOption("numRows", "nr", "Number of rows of the input matrix");
    addOption("numCols", "nc", "Number of columns of the input matrix");
    Map<String, List<String>> parsedArgs = parseArguments(strings);
    if (parsedArgs == null) {
      return -1;
    }

    int numRows = Integer.parseInt(getOption("numRows"));
    int numCols = Integer.parseInt(getOption("numCols"));

    DistributedRowMatrix matrix = new DistributedRowMatrix(getInputPath(), getTempPath(), numRows, numCols);
    matrix.setConf(new Configuration(getConf()));
    matrix.transpose();

    return 0;
  }

  public static Configuration buildTransposeJobConf(Path matrixInputPath,
                                                    Path matrixOutputPath,
                                                    int numInputRows) throws IOException {
    return buildTransposeJobConf(new Configuration(), matrixInputPath, matrixOutputPath, numInputRows);
  }

  public static Configuration buildTransposeJobConf(Configuration initialConf,
                                                    Path matrixInputPath,
                                                    Path matrixOutputPath,
                                                    int numInputRows) throws IOException {
    JobConf conf = new JobConf(initialConf, TransposeJob.class);
    conf.setJobName("TransposeJob: " + matrixInputPath + " transpose -> " + matrixOutputPath);
    FileSystem fs = FileSystem.get(matrixInputPath.toUri(), conf);
    matrixInputPath = fs.makeQualified(matrixInputPath);
    matrixOutputPath = fs.makeQualified(matrixOutputPath);
    conf.setInt(NUM_ROWS_KEY, numInputRows);

    FileInputFormat.addInputPath(conf, matrixInputPath);
    conf.setInputFormat(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(conf, matrixOutputPath);
    conf.setMapperClass(TransposeMapper.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setCombinerClass(MergeVectorsCombiner.class);
    conf.setReducerClass(MergeVectorsReducer.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    return conf;
  }

  public static class TransposeMapper extends MapReduceBase
          implements Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

    private int newNumCols;

    @Override
    public void configure(JobConf conf) {
      newNumCols = conf.getInt(NUM_ROWS_KEY, Integer.MAX_VALUE);
    }

    @Override
    public void map(IntWritable r, VectorWritable v, OutputCollector<IntWritable, VectorWritable> out,
                    Reporter reporter) throws IOException {
      int row = r.get();
      for (Vector.Element e : v.get().nonZeroes()) {
        RandomAccessSparseVector tmp = new RandomAccessSparseVector(newNumCols, 1);
        tmp.setQuick(row, e.get());
        r.set(e.index());
        out.collect(r, new VectorWritable(tmp));
      }
    }
  }

  public static class MergeVectorsCombiner extends MapReduceBase
          implements Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

    @Override
    public void reduce(WritableComparable<?> key,
                       Iterator<VectorWritable> vectors,
                       OutputCollector<WritableComparable<?>,VectorWritable> out,
                       Reporter reporter) throws IOException {
      out.collect(key, VectorWritable.merge(vectors));
    }
  }

  public static class MergeVectorsReducer extends MapReduceBase
          implements Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

    @Override
    public void reduce(WritableComparable<?> key,
                       Iterator<VectorWritable> vectors,
                       OutputCollector<WritableComparable<?>, VectorWritable> out,
                       Reporter reporter) throws IOException {
      Vector merged = VectorWritable.merge(vectors).get();
      out.collect(key, new VectorWritable(new SequentialAccessSparseVector(merged)));
    }
  }
}
