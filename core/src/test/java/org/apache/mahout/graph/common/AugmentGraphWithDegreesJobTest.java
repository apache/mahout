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

package org.apache.mahout.graph.common;

import java.io.File;
import java.util.Arrays;
import java.util.Set;

import com.google.common.collect.Sets;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.graph.model.UndirectedEdge;
import org.apache.mahout.graph.model.UndirectedEdgeWithDegrees;
import org.apache.mahout.graph.model.Vertex;
import org.apache.mahout.graph.model.VertexWithDegree;
import org.easymock.EasyMock;
import org.junit.Test;

public class AugmentGraphWithDegreesJobTest extends MahoutTestCase {

  @Test
  public void testScatterEdges() throws Exception {
    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);

    ctx.write(new Vertex(123), new Vertex(456));
    ctx.write(new Vertex(456), new Vertex(123));

    EasyMock.replay(ctx);

    new AugmentGraphWithDegreesJob.ScatterEdgesMapper()
        .map(new UndirectedEdge(new Vertex(123), new Vertex(456)), null, ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void testSumDegrees() throws Exception {
    Reducer.Context ctx = EasyMock.createMock(Reducer.Context.class);

    Vertex vertex = new Vertex(1);

    ctx.write(new UndirectedEdge(vertex, new Vertex(3)), new VertexWithDegree(vertex, 3));
    ctx.write(new UndirectedEdge(vertex, new Vertex(5)), new VertexWithDegree(vertex, 3));
    ctx.write(new UndirectedEdge(vertex, new Vertex(7)), new VertexWithDegree(vertex, 3));

    EasyMock.replay(ctx);

    new AugmentGraphWithDegreesJob.SumDegreesReducer()
        .reduce(vertex, Arrays.asList(new Vertex(3), new Vertex(5), new Vertex(7)), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void testJoinDegrees() throws Exception {
    Reducer.Context ctx = EasyMock.createMock(Reducer.Context.class);
    Vertex first = new Vertex(1);
    Vertex second = new Vertex(3);

    ctx.write(new UndirectedEdgeWithDegrees(new VertexWithDegree(first, 1), new VertexWithDegree(second, 3)),
        NullWritable.get());

    EasyMock.replay(ctx);

    new AugmentGraphWithDegreesJob.JoinDegreesReducer().reduce(new UndirectedEdge(first, second),
        Arrays.asList(new VertexWithDegree(first, 1), new VertexWithDegree(second, 3)), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void toyIntegrationTest() throws Exception {
    File inputFile = getTestTempFile("edges.seq");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tempDir = getTestTempDir("tmp");

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(inputFile.getAbsolutePath().toString()),
        UndirectedEdge.class, NullWritable.class);

    try {
      writer.append(new UndirectedEdge(0, 1), NullWritable.get());
      writer.append(new UndirectedEdge(0, 2), NullWritable.get());
      writer.append(new UndirectedEdge(0, 3), NullWritable.get());
      writer.append(new UndirectedEdge(0, 4), NullWritable.get());
      writer.append(new UndirectedEdge(0, 5), NullWritable.get());
      writer.append(new UndirectedEdge(0, 6), NullWritable.get());
      writer.append(new UndirectedEdge(0, 7), NullWritable.get());
      writer.append(new UndirectedEdge(1, 2), NullWritable.get());
      writer.append(new UndirectedEdge(1, 3), NullWritable.get());
      writer.append(new UndirectedEdge(2, 3), NullWritable.get());
      writer.append(new UndirectedEdge(4, 5), NullWritable.get());
      writer.append(new UndirectedEdge(4, 7), NullWritable.get());
    } finally {
      Closeables.closeQuietly(writer);
    }

    AugmentGraphWithDegreesJob augmentGraphWithDegreesJob = new AugmentGraphWithDegreesJob();
    augmentGraphWithDegreesJob.setConf(conf);
    augmentGraphWithDegreesJob.run(new String[] { "--input", inputFile.getAbsolutePath(),
        "--output", outputDir.getAbsolutePath(), "--tempDir", tempDir.getAbsolutePath() });

    Set<UndirectedEdgeWithDegrees> edges = Sets.newHashSet();
    for (Pair<UndirectedEdgeWithDegrees,NullWritable> result :
        new SequenceFileIterable<UndirectedEdgeWithDegrees, NullWritable>(new Path(outputDir.getAbsolutePath() +
        "/part-r-00000"), false, conf)) {
      edges.add(result.getFirst());
    }

    assertEquals(12, edges.size());
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(0, 7, 1, 3)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(0, 7, 2, 3)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(0, 7, 3, 3)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(0, 7, 4, 3)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(0, 7, 5, 2)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(0, 7, 6, 1)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(0, 7, 7, 2)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(1, 3, 2, 3)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(1, 3, 3, 3)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(2, 3, 3, 3)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(4, 3, 5, 2)));
    assertTrue(edges.contains(new UndirectedEdgeWithDegrees(4, 3, 7, 2)));
  }
}
