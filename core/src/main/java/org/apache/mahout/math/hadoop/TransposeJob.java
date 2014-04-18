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
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.mapreduce.MergeVectorsCombiner;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.common.mapreduce.TransposeMapper;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/** Transpose a matrix */
public class TransposeJob extends AbstractJob {

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

  public static Job buildTransposeJob(Path matrixInputPath, Path matrixOutputPath, int numInputRows)
    throws IOException {
    return buildTransposeJob(new Configuration(), matrixInputPath, matrixOutputPath, numInputRows);
  }

  public static Job buildTransposeJob(Configuration initialConf, Path matrixInputPath, Path matrixOutputPath,
      int numInputRows) throws IOException {

    Job job = HadoopUtil.prepareJob(matrixInputPath, matrixOutputPath, SequenceFileInputFormat.class,
        TransposeMapper.class, IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class,
        VectorWritable.class, SequenceFileOutputFormat.class, initialConf);
    job.setCombinerClass(MergeVectorsCombiner.class);
    job.getConfiguration().setInt(TransposeMapper.NEW_NUM_COLS_PARAM, numInputRows);

    job.setJobName("TransposeJob: " + matrixInputPath);

    return job;
  }


}
