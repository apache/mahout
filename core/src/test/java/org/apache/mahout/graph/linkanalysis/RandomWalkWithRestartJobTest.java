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
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.graph.GraphTestCase;
import org.apache.mahout.graph.preprocessing.GraphUtils;
import org.apache.mahout.graph.model.Edge;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Test;

import java.io.File;
import java.util.Map;

public class RandomWalkWithRestartJobTest extends GraphTestCase {

  @Test
  public void toyIntegrationTest() throws Exception {

    File verticesFile = getTestTempFile("vertices.txt");
    File edgesFile = getTestTempFile("edges.seq");
    File indexedVerticesFile = getTestTempFile("indexedVertices.seq");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tempDir = getTestTempDir();

    Configuration conf = new Configuration();

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

    int numVertices = GraphUtils.indexVertices(conf, new Path(verticesFile.getAbsolutePath()),
        new Path(indexedVerticesFile.getAbsolutePath()));

    RandomWalkWithRestartJob randomWalkWithRestart = new RandomWalkWithRestartJob();
    randomWalkWithRestart.setConf(conf);
    randomWalkWithRestart.run(new String[]{"--vertexIndex", indexedVerticesFile.getAbsolutePath(),
        "--edges", edgesFile.getAbsolutePath(), "--sourceVertexIndex", String.valueOf(2),
        "--output", outputDir.getAbsolutePath(), "--numVertices", String.valueOf(numVertices),
        "--numIterations", String.valueOf(2), "--stayingProbability", String.valueOf(0.75),
        "--tempDir", tempDir.getAbsolutePath()});

    Matrix expectedAdjacenyMatrix = new DenseMatrix(new double[][] {
        { 0,     0,     0.1875, 0    },
        { 0.375, 0.375, 0.1875, 0.75 },
        { 0.375, 0,     0.1875, 0    },
        { 0,     0.375, 0.1875, 0    } });

    Matrix actualAdjacencyMatrix = MathHelper.readMatrix(conf, new Path(tempDir.getAbsolutePath(),
        "adjacencyMatrix/part-r-00000"), numVertices, numVertices);

    assertMatrixEquals(expectedAdjacenyMatrix, actualAdjacencyMatrix);

    Map<Long,Double> steadyStateProbabilities = Maps.newHashMap();
    for (String line : new FileLineIterable(new File(outputDir, "part-m-00000"))) {
      String[] tokens = Iterables.toArray(Splitter.on("\t").split(line), String.class);
      steadyStateProbabilities.put(Long.parseLong(tokens[0]), Double.parseDouble(tokens[1]));
    }

    assertEquals(4, steadyStateProbabilities.size());

    assertEquals(steadyStateProbabilities.get(12L), 75.0 / 1024.0, EPSILON);
    assertEquals(steadyStateProbabilities.get(34L), 363.0 / 1024.0, EPSILON);
    assertEquals(steadyStateProbabilities.get(56L), 349.0 / 1024.0, EPSILON);
    assertEquals(steadyStateProbabilities.get(78L), 237.0 / 1024.0, EPSILON);
  }

}
