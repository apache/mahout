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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.graph.GraphTestCase;
import org.apache.mahout.graph.common.GraphUtils;
import org.apache.mahout.graph.model.Edge;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.hadoop.MathHelper;
import org.apache.mahout.math.map.OpenLongIntHashMap;
import org.easymock.EasyMock;
import org.junit.Test;

import java.io.File;
import java.util.Map;

/** example from "Mining Massive Datasets" */
public class PageRankJobTest extends GraphTestCase {

  @Test
  public void indexAndCountDegree() throws Exception {

    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);

    ctx.write(new IntWritable(7), new IntWritable(1));

    EasyMock.replay(ctx);

    OpenLongIntHashMap index = new OpenLongIntHashMap();
    index.put(123L, 7);
    PageRankJob.IndexAndCountDegreeMapper indexAndCountDegreeMapper = new PageRankJob.IndexAndCountDegreeMapper();
    setField(indexAndCountDegreeMapper, "vertexIDsToIndex", index);
    indexAndCountDegreeMapper.map(new Edge(123L, 456L), NullWritable.get(), ctx);

    EasyMock.verify(ctx);
  }

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
        new Edge(12, 78),
        new Edge(34, 12),
        new Edge(34, 78),
        new Edge(56, 56),
        new Edge(78, 34),
        new Edge(78, 56));

    int numVertices = GraphUtils.indexVertices(conf, new Path(verticesFile.getAbsolutePath()),
        new Path(indexedVerticesFile.getAbsolutePath()));

    PageRankJob pageRank = new PageRankJob();
    pageRank.setConf(conf);
    pageRank.run(new String[] { "--vertexIndex", indexedVerticesFile.getAbsolutePath(),
        "--edges", edgesFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--numVertices", String.valueOf(numVertices), "--numIterations", String.valueOf(3),
        "--teleportationProbability", String.valueOf(0.8), "--tempDir", tempDir.getAbsolutePath() });

    DenseMatrix expectedTransitionMatrix = new DenseMatrix(new double[][]{
        { 0,           0.4, 0,   0 },
        { 0.266666667, 0,   0,   0.4 },
        { 0.266666667, 0,   0.8, 0.4 },
        { 0.266666667, 0.4, 0,   0 } });

    Matrix actualTransitionMatrix = MathHelper.readEntries(conf, new Path(tempDir.getAbsolutePath(),
        "transitionMatrix/part-r-00000"), numVertices, numVertices);

    assertEquals(expectedTransitionMatrix, actualTransitionMatrix);

    Map<Long,Double> rankPerVertex = Maps.newHashMap();
    for (String line : new FileLineIterable(new File(outputDir, "part-r-00000"))) {
      String[] tokens = Iterables.toArray(Splitter.on("\t").split(line), String.class);
      rankPerVertex.put(Long.parseLong(tokens[0]), Double.parseDouble(tokens[1]));
    }

    assertEquals(4, rankPerVertex.size());
    assertEquals(rankPerVertex.get(12l), 0.1206666, EPSILON);
    assertEquals(rankPerVertex.get(34L), 0.1571111, EPSILON);
    assertEquals(rankPerVertex.get(56L), 0.5651111, EPSILON);
    assertEquals(rankPerVertex.get(78L), 0.1571111, EPSILON);

  }

  void assertEquals(Matrix expected, Matrix actual) {
    assertEquals(expected.numRows(), actual.numRows());
    assertEquals(actual.numCols(), actual.numCols());
    for (int row = 0; row < expected.numRows(); row++) {
      for (int col = 0; col < expected.numCols(); col ++) {
        assertEquals("Non-matching values in [" + row + "," + col + "]",
            expected.get(row, col), actual.get(row, col), EPSILON);
      }
    }
  }

}
