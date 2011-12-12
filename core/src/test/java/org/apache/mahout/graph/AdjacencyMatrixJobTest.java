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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class AdjacencyMatrixJobTest extends MahoutTestCase {

  private static final Logger log = LoggerFactory.getLogger(AdjacencyMatrixJobTest.class);

  @Test
  public void adjacencyMatrix() throws Exception {
    File verticesFile = getTestTempFile("vertices.txt");
    File edgesFile = getTestTempFile("edges.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();

    Configuration conf = new Configuration();

    writeLines(verticesFile, "12", "34", "56", "78");

    writeLines(edgesFile, 
        "12,34",
        "12,56",
        "34,34",
        "34,78",
        "56,12",
        "56,34",
        "56,56",
        "56,78",
        "78,34");

    Matrix expectedAdjacencyMatrix = new DenseMatrix(new double[][] {
        { 0, 1, 1, 0 },
        { 0, 1, 0, 1 },
        { 1, 1, 1, 1 },
        { 0, 1, 0, 0 } });

    AdjacencyMatrixJob createAdjacencyMatrix = new AdjacencyMatrixJob();
    createAdjacencyMatrix.setConf(conf);
    createAdjacencyMatrix.run(new String[] { "--vertices", verticesFile.getAbsolutePath(),
        "--edges", edgesFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath() });

    int numVertices = HadoopUtil.readInt(new Path(outputDir.getAbsolutePath(), AdjacencyMatrixJob.NUM_VERTICES), conf);
    Matrix actualAdjacencyMatrix = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(),
        AdjacencyMatrixJob.ADJACENCY_MATRIX + "/part-r-00000"), numVertices, numVertices);

    StringBuilder info = new StringBuilder();
    info.append("expected adjacency matrix:\n");
    info.append(MathHelper.nice(expectedAdjacencyMatrix));
    info.append("actual adjacency matrix:\n");
    info.append(MathHelper.nice(actualAdjacencyMatrix));

    log.info(info.toString());

    MathHelper.assertMatrixEquals(expectedAdjacencyMatrix, actualAdjacencyMatrix);
  }

  @Test
  public void symmetricAdjacencyMatrix() throws Exception {
    File verticesFile = getTestTempFile("vertices.txt");
    File edgesFile = getTestTempFile("edges.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();

    Configuration conf = new Configuration();

    writeLines(verticesFile, "12", "34", "56", "78");

    writeLines(edgesFile,
        "12,34",
        "12,56",
        "34,34",
        "34,78",
        "56,34",
        "56,56",
        "56,78");

    Matrix expectedAdjacencyMatrix = new DenseMatrix(new double[][] {
        { 0, 1, 1, 0 },
        { 1, 1, 1, 1 },
        { 1, 1, 1, 1 },
        { 0, 1, 1, 0 } });

    AdjacencyMatrixJob createAdjacencyMatrix = new AdjacencyMatrixJob();
    createAdjacencyMatrix.setConf(conf);
    createAdjacencyMatrix.run(new String[] { "--vertices", verticesFile.getAbsolutePath(),
        "--edges", edgesFile.getAbsolutePath(), "--symmetric", String.valueOf(true),
        "--output", outputDir.getAbsolutePath() });

    int numVertices = HadoopUtil.readInt(new Path(outputDir.getAbsolutePath(), AdjacencyMatrixJob.NUM_VERTICES), conf);
    Matrix actualAdjacencyMatrix = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(),
        AdjacencyMatrixJob.ADJACENCY_MATRIX + "/part-r-00000"), numVertices, numVertices);

    StringBuilder info = new StringBuilder();
    info.append("expected adjacency matrix:\n");
    info.append(MathHelper.nice(expectedAdjacencyMatrix));
    info.append("actual adjacency matrix:\n");
    info.append(MathHelper.nice(actualAdjacencyMatrix));

    log.info(info.toString());

    MathHelper.assertMatrixEquals(expectedAdjacencyMatrix, actualAdjacencyMatrix);
  }
}