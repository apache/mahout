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

package org.apache.mahout.math.hadoop.solver;

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.TestDistributedRowMatrix;
import org.junit.Test;

public final class TestDistributedConjugateGradientSolverCLI extends MahoutTestCase {

  private static Vector randomVector(int size, double entryMean) {
    Vector v = new DenseVector(size);
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < size; ++i) {
      v.setQuick(i, r.nextGaussian() * entryMean);
    }
    return v;
  }

  private static Path saveVector(Configuration conf, Path path, Vector v) throws IOException {
    FileSystem fs = path.getFileSystem(conf);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, IntWritable.class, VectorWritable.class);
    
    try {
      writer.append(new IntWritable(0), new VectorWritable(v));
    } finally {
      writer.close();
    }
    return path;
  }
  
  private static Vector loadVector(Configuration conf, Path path) throws IOException {
    FileSystem fs = path.getFileSystem(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    Writable key = new IntWritable();
    VectorWritable value = new VectorWritable();
    
    try {
      if (!reader.next(key, value)) {
        throw new IOException("Input vector file is empty.");      
      }
      return value.get();
    } finally {
      reader.close();
    }
  }

  @Test
  public void testSolver() throws Exception {
    Configuration conf = getConfiguration();
    Path testData = getTestTempDirPath("testdata");
    DistributedRowMatrix matrix = new TestDistributedRowMatrix().randomDistributedMatrix(
        10, 10, 10, 10, 10.0, true, testData.toString());
    matrix.setConf(conf);
    Path output = getTestTempFilePath("output");
    Path vectorPath = getTestTempFilePath("vector");
    Path tempPath = getTestTempDirPath("tmp");

    Vector vector = randomVector(matrix.numCols(), 10.0);
    saveVector(conf, vectorPath, vector);
        
    String[] args = {
        "-i", matrix.getRowPath().toString(),
        "-o", output.toString(),
        "--tempDir", tempPath.toString(),
        "--vector", vectorPath.toString(),
        "--numRows", "10",
        "--numCols", "10",
        "--symmetric", "true"        
    };
    
    DistributedConjugateGradientSolver solver = new DistributedConjugateGradientSolver();
    ToolRunner.run(getConfiguration(), solver.job(), args);
    
    Vector x = loadVector(conf, output);
    
    Vector solvedVector = matrix.times(x);    
    double distance = Math.sqrt(vector.getDistanceSquared(solvedVector));
    assertEquals(0.0, distance, EPSILON);
  }
}
