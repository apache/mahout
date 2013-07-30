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

package org.apache.mahout.math.hadoop.similarity.cooccurrence;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.hadoop.MathHelper;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.TanimotoCoefficientSimilarity;
import org.apache.mahout.math.map.OpenIntIntHashMap;
import org.junit.Test;

import java.io.File;

public class RowSimilarityJobTest extends MahoutTestCase {

  /**
   * integration test with a tiny data set
   *
   * <pre>
   *
   * input matrix:
   *
   * 1, 0, 1, 1, 0
   * 0, 0, 1, 1, 0
   * 0, 0, 0, 0, 1
   *
   * similarity matrix (via tanimoto):
   *
   * 1,     0.666, 0
   * 0.666, 1,     0
   * 0,     0,     1
   * </pre>
   * @throws Exception
   */
  @Test
  public void toyIntegration() throws Exception {

    File inputFile = getTestTempFile("rows");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    Configuration conf = getConfiguration();
    Path inputPath = new Path(inputFile.getAbsolutePath());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    MathHelper.writeDistributedRowMatrix(new double[][] {
        new double[] { 1, 0, 1, 1, 0 },
        new double[] { 0, 0, 1, 1, 0 },
        new double[] { 0, 0, 0, 0, 1 } },
        fs, conf, inputPath);

    RowSimilarityJob rowSimilarityJob = new RowSimilarityJob();
    rowSimilarityJob.setConf(conf);
    rowSimilarityJob.run(new String[] { "--input", inputFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--numberOfColumns", String.valueOf(5), "--similarityClassname", TanimotoCoefficientSimilarity.class.getName(),
        "--tempDir", tmpDir.getAbsolutePath() });


    OpenIntIntHashMap observationsPerColumn =
        Vectors.readAsIntMap(new Path(tmpDir.getAbsolutePath(), "observationsPerColumn.bin"), conf);
    assertEquals(4, observationsPerColumn.size());
    assertEquals(1, observationsPerColumn.get(0));
    assertEquals(2, observationsPerColumn.get(2));
    assertEquals(2, observationsPerColumn.get(3));
    assertEquals(1, observationsPerColumn.get(4));

    Matrix similarityMatrix = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "part-r-00000"), 3, 3);

    assertNotNull(similarityMatrix);
    assertEquals(3, similarityMatrix.numCols());
    assertEquals(3, similarityMatrix.numRows());

    assertEquals(1.0, similarityMatrix.get(0, 0), EPSILON);
    assertEquals(1.0, similarityMatrix.get(1, 1), EPSILON);
    assertEquals(1.0, similarityMatrix.get(2, 2), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 0), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(0, 2), EPSILON);
    assertEquals(0.0, similarityMatrix.get(1, 2), EPSILON);
    assertEquals(0.666666, similarityMatrix.get(0, 1), EPSILON);
    assertEquals(0.666666, similarityMatrix.get(1, 0), EPSILON);
  }

  @Test
  public void toyIntegrationMaxSimilaritiesPerRow() throws Exception {

    File inputFile = getTestTempFile("rows");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    Configuration conf = getConfiguration();
    Path inputPath = new Path(inputFile.getAbsolutePath());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    MathHelper.writeDistributedRowMatrix(new double[][]{
        new double[] { 1, 0, 1, 1, 0, 1 },
        new double[] { 0, 1, 1, 1, 1, 1 },
        new double[] { 1, 1, 0, 1, 0, 0 } },
        fs, conf, inputPath);

    RowSimilarityJob rowSimilarityJob = new RowSimilarityJob();
    rowSimilarityJob.setConf(conf);
    rowSimilarityJob.run(new String[] { "--input", inputFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--numberOfColumns", String.valueOf(6), "--similarityClassname", TanimotoCoefficientSimilarity.class.getName(),
        "--maxSimilaritiesPerRow", String.valueOf(1), "--excludeSelfSimilarity", String.valueOf(true),
        "--tempDir", tmpDir.getAbsolutePath() });

    Matrix similarityMatrix = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "part-r-00000"), 3, 3);

    assertNotNull(similarityMatrix);
    assertEquals(3, similarityMatrix.numCols());
    assertEquals(3, similarityMatrix.numRows());

    assertEquals(0.0, similarityMatrix.get(0, 0), EPSILON);
    assertEquals(0.5, similarityMatrix.get(0, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(0, 2), EPSILON);

    assertEquals(0.5, similarityMatrix.get(1, 0), EPSILON);
    assertEquals(0.0, similarityMatrix.get(1, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(1, 2), EPSILON);

    assertEquals(0.4, similarityMatrix.get(2, 0), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 2), EPSILON);
  }

  @Test
  public void toyIntegrationWithThreshold() throws Exception {


    File inputFile = getTestTempFile("rows");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    Configuration conf = getConfiguration();
    Path inputPath = new Path(inputFile.getAbsolutePath());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    MathHelper.writeDistributedRowMatrix(new double[][]{
        new double[] { 1, 0, 1, 1, 0, 1 },
        new double[] { 0, 1, 1, 1, 1, 1 },
        new double[] { 1, 1, 0, 1, 0, 0 } },
        fs, conf, inputPath);

    RowSimilarityJob rowSimilarityJob = new RowSimilarityJob();
    rowSimilarityJob.setConf(conf);
    rowSimilarityJob.run(new String[] { "--input", inputFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--numberOfColumns", String.valueOf(6), "--similarityClassname", TanimotoCoefficientSimilarity.class.getName(),
        "--excludeSelfSimilarity", String.valueOf(true), "--threshold", String.valueOf(0.5),
        "--tempDir", tmpDir.getAbsolutePath() });

    Matrix similarityMatrix = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "part-r-00000"), 3, 3);

    assertNotNull(similarityMatrix);
    assertEquals(3, similarityMatrix.numCols());
    assertEquals(3, similarityMatrix.numRows());

    assertEquals(0.0, similarityMatrix.get(0, 0), EPSILON);
    assertEquals(0.5, similarityMatrix.get(0, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(0, 2), EPSILON);

    assertEquals(0.5, similarityMatrix.get(1, 0), EPSILON);
    assertEquals(0.0, similarityMatrix.get(1, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(1, 2), EPSILON);

    assertEquals(0.0, similarityMatrix.get(2, 0), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 1), EPSILON);
    assertEquals(0.0, similarityMatrix.get(2, 2), EPSILON);
  }

  @Test
  public void testVectorDimensions() throws Exception {

    File inputFile = getTestTempFile("rows");

    Configuration conf = getConfiguration();
    Path inputPath = new Path(inputFile.getAbsolutePath());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    MathHelper.writeDistributedRowMatrix(new double[][] {
        new double[] { 1, 0, 1, 1, 0, 1 },
        new double[] { 0, 1, 1, 1, 1, 1 },
        new double[] { 1, 1, 0, 1, 0, 0 } },
        fs, conf, inputPath);

    RowSimilarityJob rowSimilarityJob = new RowSimilarityJob();
    rowSimilarityJob.setConf(conf);

    int numberOfColumns = rowSimilarityJob.getDimensions(inputPath);

    assertEquals(6, numberOfColumns);
  }
}
