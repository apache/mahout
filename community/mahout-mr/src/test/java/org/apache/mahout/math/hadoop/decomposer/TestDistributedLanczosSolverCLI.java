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

package org.apache.mahout.math.hadoop.decomposer;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.TestDistributedRowMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Arrays;

@Deprecated
public final class TestDistributedLanczosSolverCLI extends MahoutTestCase {
  private static final Logger log = LoggerFactory.getLogger(TestDistributedLanczosSolverCLI.class);

  @Test
  public void testDistributedLanczosSolverCLI() throws Exception {
    Path testData = getTestTempDirPath("testdata");
    DistributedRowMatrix corpus =
        new TestDistributedRowMatrix().randomDenseHierarchicalDistributedMatrix(10, 9, false,
            testData.toString());
    corpus.setConf(getConfiguration());
    Path output = getTestTempDirPath("output");
    Path tmp = getTestTempDirPath("tmp");
    Path workingDir = getTestTempDirPath("working");
    String[] args = {
        "-i", new Path(testData, "distMatrix").toString(),
        "-o", output.toString(),
        "--tempDir", tmp.toString(),
        "--numRows", "10",
        "--numCols", "9",
        "--rank", "6",
        "--symmetric", "false",
        "--workingDir", workingDir.toString()
    };
    ToolRunner.run(getConfiguration(), new DistributedLanczosSolver().new DistributedLanczosSolverJob(), args);

    output = getTestTempDirPath("output2");
    tmp = getTestTempDirPath("tmp2");
    args = new String[] {
        "-i", new Path(testData, "distMatrix").toString(),
        "-o", output.toString(),
        "--tempDir", tmp.toString(),
        "--numRows", "10",
        "--numCols", "9",
        "--rank", "7",
        "--symmetric", "false",
        "--workingDir", workingDir.toString()
    };
    ToolRunner.run(getConfiguration(), new DistributedLanczosSolver().new DistributedLanczosSolverJob(), args);

    Path rawEigenvectors = new Path(output, DistributedLanczosSolver.RAW_EIGENVECTORS);
    Matrix eigenVectors = new DenseMatrix(7, corpus.numCols());
    Configuration conf = getConfiguration();

    int i = 0;
    for (VectorWritable value : new SequenceFileValueIterable<VectorWritable>(rawEigenvectors, conf)) {
      Vector v = value.get();
      eigenVectors.assignRow(i, v);
      i++;
    }
    assertEquals("number of eigenvectors", 7, i);
  }

  @Test
  public void testDistributedLanczosSolverEVJCLI() throws Exception {
    Path testData = getTestTempDirPath("testdata");
    DistributedRowMatrix corpus = new TestDistributedRowMatrix()
        .randomDenseHierarchicalDistributedMatrix(10, 9, false, testData.toString());
    corpus.setConf(getConfiguration());
    Path output = getTestTempDirPath("output");
    Path tmp = getTestTempDirPath("tmp");
    String[] args = {
        "-i", new Path(testData, "distMatrix").toString(),
        "-o", output.toString(),
        "--tempDir", tmp.toString(),
        "--numRows", "10",
        "--numCols", "9",
        "--rank", "6",
        "--symmetric", "false",
        "--cleansvd", "true"
    };
    ToolRunner.run(getConfiguration(), new DistributedLanczosSolver().new DistributedLanczosSolverJob(), args);
  
    Path cleanEigenvectors = new Path(output, EigenVerificationJob.CLEAN_EIGENVECTORS);
    Matrix eigenVectors = new DenseMatrix(6, corpus.numCols());
    Collection<Double> eigenvalues = Lists.newArrayList();

    output = getTestTempDirPath("output2");
    tmp = getTestTempDirPath("tmp2");
    args = new String[] {
        "-i", new Path(testData, "distMatrix").toString(),
        "-o", output.toString(),
        "--tempDir", tmp.toString(),
        "--numRows", "10",
        "--numCols", "9",
        "--rank", "7",
        "--symmetric", "false",
        "--cleansvd", "true"
    };
    ToolRunner.run(getConfiguration(), new DistributedLanczosSolver().new DistributedLanczosSolverJob(), args);
    Path cleanEigenvectors2 = new Path(output, EigenVerificationJob.CLEAN_EIGENVECTORS);
    Matrix eigenVectors2 = new DenseMatrix(7, corpus.numCols());
    Configuration conf = getConfiguration();
    Collection<Double> newEigenValues = Lists.newArrayList();

    int i = 0;
    for (VectorWritable value : new SequenceFileValueIterable<VectorWritable>(cleanEigenvectors, conf)) {
      NamedVector v = (NamedVector) value.get();
      eigenVectors.assignRow(i, v);
      log.info(v.getName());
      if (EigenVector.getCosAngleError(v.getName()) < 1.0e-3) {
        eigenvalues.add(EigenVector.getEigenValue(v.getName()));
      }
      i++;
    }
    assertEquals("number of clean eigenvectors", 3, i);

    i = 0;
    for (VectorWritable value : new SequenceFileValueIterable<VectorWritable>(cleanEigenvectors2, conf)) {
      NamedVector v = (NamedVector) value.get();
      log.info(v.getName());
      eigenVectors2.assignRow(i, v);
      newEigenValues.add(EigenVector.getEigenValue(v.getName()));
      i++;
    }

    Collection<Integer> oldEigensFound = Lists.newArrayList();
    for (int row = 0; row < eigenVectors.numRows(); row++) {
      Vector oldEigen = eigenVectors.viewRow(row);
      if (oldEigen == null) {
        break;
      }
      for (int newRow = 0; newRow < eigenVectors2.numRows(); newRow++) {
        Vector newEigen = eigenVectors2.viewRow(newRow);
        if (newEigen != null && oldEigen.dot(newEigen) > 0.9) {
          oldEigensFound.add(row);
          break;
        }
      }
    }
    assertEquals("the number of new eigenvectors", 5, i);

    Collection<Double> oldEigenValuesNotFound = Lists.newArrayList();
    for (double d : eigenvalues) {
      boolean found = false;
      for (double newD : newEigenValues) {
        if (Math.abs((d - newD)/d) < 0.1) {
          found = true;
        }
      }
      if (!found) {
        oldEigenValuesNotFound.add(d);
      }
    }
    assertEquals("number of old eigenvalues not found: "
                 + Arrays.toString(oldEigenValuesNotFound.toArray(new Double[oldEigenValuesNotFound.size()])),
                0, oldEigenValuesNotFound.size());
    assertEquals("did not find enough old eigenvectors", 3, oldEigensFound.size());
  }

}
