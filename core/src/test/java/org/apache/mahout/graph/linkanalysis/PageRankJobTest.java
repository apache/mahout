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

package org.apache.mahout.graph.linkanalysis;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.graph.AdjacencyMatrixJob;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Map;

/** example from "Mining Massive Datasets" (page 157) */
public final class PageRankJobTest extends MahoutTestCase {

  private static final Logger log = LoggerFactory.getLogger(PageRankJobTest.class);
  @Test
  public void toyIntegrationTest() throws Exception {

    File verticesFile = getTestTempFile("vertices.txt");
    File edgesFile = getTestTempFile("edges.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tempDir = getTestTempDir();

    Configuration conf = new Configuration();

    writeLines(verticesFile, "12", "34", "56", "78");

    writeLines(edgesFile,
        "12,34",
        "12,56",
        "12,78",
        "34,12",
        "34,78",
        "56,56",
        "78,34",
        "78,56");

    PageRankJob pageRank = new PageRankJob();
    pageRank.setConf(conf);
    pageRank.run(new String[] { "--vertices", verticesFile.getAbsolutePath(), "--edges", edgesFile.getAbsolutePath(),
        "--output", outputDir.getAbsolutePath(), "--numIterations", "3", "--stayingProbability", "0.8",
        "--tempDir", tempDir.getAbsolutePath() });

    Matrix expectedAdjacencyMatrix = new DenseMatrix(new double[][] {
        { 0, 1, 1, 1 },
        { 1, 0, 0, 1 },
        { 0, 0, 1, 0 },
        { 0, 1, 1, 0 } });

    int numVertices = HadoopUtil.readInt(new Path(tempDir.getAbsolutePath(), AdjacencyMatrixJob.NUM_VERTICES), conf);
    assertEquals(4, numVertices);
    Matrix actualAdjacencyMatrix = MathHelper.readMatrix(conf, new Path(tempDir.getAbsolutePath(),
        AdjacencyMatrixJob.ADJACENCY_MATRIX + "/part-r-00000"), numVertices, numVertices);

    StringBuilder info = new StringBuilder();
    info.append("\nexpected adjacency matrix\n\n");
    info.append(MathHelper.nice(expectedAdjacencyMatrix));
    info.append("\nactual adjacency matrix \n\n");
    info.append(MathHelper.nice(actualAdjacencyMatrix));
    info.append('\n');
    log.info(info.toString());

    Matrix expectedTransitionMatrix = new DenseMatrix(new double[][] {
        { 0.0,         0.4, 0.0, 0.0 },
        { 0.266666667, 0.0, 0.0, 0.4 },
        { 0.266666667, 0.0, 0.8, 0.4 },
        { 0.266666667, 0.4, 0.0, 0.0 } });

    Matrix actualTransitionMatrix = MathHelper.readMatrix(conf, new Path(tempDir.getAbsolutePath(),
        "transitionMatrix/part-r-00000"), numVertices, numVertices);

    info = new StringBuilder();
    info.append("\nexpected transition matrix\n\n");
    info.append(MathHelper.nice(expectedTransitionMatrix));
    info.append("\nactual transition matrix\n\n");
    info.append(MathHelper.nice(actualTransitionMatrix));
    info.append('\n');
    log.info(info.toString());

    MathHelper.assertMatrixEquals(expectedAdjacencyMatrix, actualAdjacencyMatrix);
    MathHelper.assertMatrixEquals(expectedTransitionMatrix, actualTransitionMatrix);

    Map<Integer,Double> rankPerVertex = Maps.newHashMapWithExpectedSize(numVertices);
    for (CharSequence line : new FileLineIterable(new File(outputDir, "part-m-00000"))) {
      String[] tokens = Iterables.toArray(Splitter.on("\t").split(line), String.class);
      rankPerVertex.put(Integer.parseInt(tokens[0]), Double.parseDouble(tokens[1]));
    }

    assertEquals(4, rankPerVertex.size());
    assertEquals(0.1206666, rankPerVertex.get(12), EPSILON);
    assertEquals(0.1571111, rankPerVertex.get(34), EPSILON);
    assertEquals(0.5651111, rankPerVertex.get(56), EPSILON);
    assertEquals(0.1571111, rankPerVertex.get(78), EPSILON);
  }


}
