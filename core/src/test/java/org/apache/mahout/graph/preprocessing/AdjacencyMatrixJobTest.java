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

package org.apache.mahout.graph.preprocessing;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.graph.GraphTestCase;
import org.apache.mahout.graph.model.Edge;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Before;
import org.junit.Test;

import java.io.File;

public class AdjacencyMatrixJobTest extends GraphTestCase {

  private File edgesFile;
  private File indexedVerticesFile;
  private File outputDir;
  private File tempDir;
  private int numVertices;
  private double stayingProbability;
  private Matrix expectedAdjacencyMatrix;
  private Configuration conf;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    File verticesFile = getTestTempFile("vertices.txt");
    edgesFile = getTestTempFile("edges.seq");
    indexedVerticesFile = getTestTempFile("indexedVertices.seq");
    outputDir = getTestTempDir("output");
    outputDir.delete();
    tempDir = getTestTempDir();

    conf = new Configuration();

    writeLines(verticesFile, "12", "34", "56", "78");

    writeComponents(edgesFile, conf, Edge.class,
        new Edge(12, 34),
        new Edge(12, 56),
        new Edge(34, 34),
        new Edge(34, 78),
        new Edge(56, 12),
        new Edge(56, 34),
        new Edge(56, 56),
        new Edge(56, 78),
        new Edge(78, 34));

    numVertices = GraphUtils.indexVertices(conf, new Path(verticesFile.getAbsolutePath()),
        new Path(indexedVerticesFile.getAbsolutePath()));

    expectedAdjacencyMatrix = new DenseMatrix(new double[][] {
        { 0, 0, 1, 0 },
        { 1, 1, 1, 1 },
        { 1, 0, 1, 0 },
        { 0, 1, 1, 0 } });

    stayingProbability = 0.5;
  }

  @Test
  public void adjacencyMatrix() throws Exception {
    AdjacencyMatrixJob createAdjacencyMatrix = new AdjacencyMatrixJob();
    createAdjacencyMatrix.setConf(conf);
    createAdjacencyMatrix.run(new String[] { "--vertexIndex", indexedVerticesFile.getAbsolutePath(),
        "--edges", edgesFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--numVertices", String.valueOf(numVertices), "--tempDir", tempDir.getAbsolutePath() });

    Matrix actualAdjacencyMatrix = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "part-r-00000"),
        numVertices, numVertices);
    assertMatrixEquals(expectedAdjacencyMatrix, actualAdjacencyMatrix);
  }

  @Test
  public void substochastifiedAdjacencyMatrix() throws Exception {
    AdjacencyMatrixJob createAdjacencyMatrix = new AdjacencyMatrixJob();
    createAdjacencyMatrix.setConf(conf);
    createAdjacencyMatrix.run(new String[] { "--vertexIndex", indexedVerticesFile.getAbsolutePath(),
        "--edges", edgesFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--numVertices", String.valueOf(numVertices), "--substochastify", String.valueOf(true),
        "--tempDir", tempDir.getAbsolutePath() });

    Matrix actualAdjacencyMatrix = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "part-r-00000"),
        numVertices, numVertices);

    substochastifyExpectedAdjacencyMatrix();

    assertMatrixEquals(expectedAdjacencyMatrix, actualAdjacencyMatrix);
  }

  @Test
  public void substochasitifiedAdjacencyMatrixWithTeleports() throws Exception {
    AdjacencyMatrixJob createAdjacencyMatrix = new AdjacencyMatrixJob();
    createAdjacencyMatrix.setConf(conf);
    createAdjacencyMatrix.run(new String[] { "--vertexIndex", indexedVerticesFile.getAbsolutePath(),
        "--edges", edgesFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--numVertices", String.valueOf(numVertices), "--substochastify", String.valueOf(true),
        "--stayingProbability", String.valueOf(stayingProbability), "--tempDir", tempDir.getAbsolutePath() });

    Matrix actualAdjacencyMatrix = MathHelper.readMatrix(conf, new Path(outputDir.getAbsolutePath(), "part-r-00000"),
        numVertices, numVertices);

    substochastifyExpectedAdjacencyMatrix();
    adjustExpectedAdjacencyMatrixForTeleports();

    assertMatrixEquals(expectedAdjacencyMatrix, actualAdjacencyMatrix);
  }

  private void substochastifyExpectedAdjacencyMatrix() {
    for (int column = 0; column < expectedAdjacencyMatrix.numCols(); column++) {
      expectedAdjacencyMatrix.assignColumn(column, expectedAdjacencyMatrix.viewColumn(column).normalize(1));
    }
  }

  private void adjustExpectedAdjacencyMatrixForTeleports() {
    expectedAdjacencyMatrix.assign(new DoubleFunction() {
      @Override
      public double apply(double val) {
        return val * stayingProbability;
      }
    });
  }

}