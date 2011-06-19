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

package org.apache.mahout.graph.triangles;

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
import org.apache.mahout.graph.model.Triangle;
import org.apache.mahout.graph.model.UndirectedEdgeWithDegrees;
import org.apache.mahout.graph.model.Vertex;
import org.easymock.EasyMock;
import org.junit.Test;

public class EnumerateTrianglesJobTest extends MahoutTestCase {

  @Test
  public void testScatterEdgesToLowerVertexDegree() throws Exception {
    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);

    ctx.write(new Vertex(1), new Vertex(3));

    EasyMock.replay(ctx);

    new EnumerateTrianglesJob.ScatterEdgesToLowerDegreeVertexMapper()
        .map(new UndirectedEdgeWithDegrees(1, 5, 3, 7), null, ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void testBuildOpenTriads() throws Exception {
    Reducer.Context ctx = EasyMock.createMock(Reducer.Context.class);

    ctx.write(new JoinableUndirectedEdge(1, 2, false), new VertexOrMarker(0));
    ctx.write(new JoinableUndirectedEdge(1, 3, false), new VertexOrMarker(0));
    ctx.write(new JoinableUndirectedEdge(2, 3, false), new VertexOrMarker(0));

    EasyMock.replay(ctx);

    new EnumerateTrianglesJob.BuildOpenTriadsReducer().reduce(new Vertex(0), Arrays.asList(new Vertex(1), new Vertex(2),
        new Vertex(3)), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void testJoinTriangles() throws Exception {
    Reducer.Context ctx = EasyMock.createMock(Reducer.Context.class);

    ctx.write(new Triangle(0, 1, 2), NullWritable.get());
    ctx.write(new Triangle(0, 2, 3), NullWritable.get());

    EasyMock.replay(ctx);

    new EnumerateTrianglesJob.JoinTrianglesReducer().reduce(new JoinableUndirectedEdge(0, 2, true),
        Arrays.asList(VertexOrMarker.MARKER, new VertexOrMarker(1), new VertexOrMarker(3)), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void testJoinTrianglesNoTriangleIfMarkerIsMissing() throws Exception {
    Reducer.Context ctx = EasyMock.createMock(Reducer.Context.class);

    EasyMock.replay(ctx);

    new EnumerateTrianglesJob.JoinTrianglesReducer().reduce(new JoinableUndirectedEdge(0, 2, false),
        Arrays.asList(new VertexOrMarker(1), new VertexOrMarker(3)), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void toyIntegrationTest() throws Exception {
    File inputFile = getTestTempFile("edgesWithDegrees.seq");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tempDir = getTestTempDir("tmp");

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(inputFile.getAbsolutePath().toString()),
        UndirectedEdgeWithDegrees.class, NullWritable.class);

    try {
      writer.append(new UndirectedEdgeWithDegrees(0, 7, 1, 3), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(0, 7, 2, 3), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(0, 7, 3, 3), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(0, 7, 4, 3), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(0, 7, 5, 2), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(0, 7, 6, 1), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(0, 7, 7, 2), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(1, 3, 2, 3), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(1, 3, 3, 3), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(2, 3, 3, 3), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(4, 3, 5, 2), NullWritable.get());
      writer.append(new UndirectedEdgeWithDegrees(4, 3, 7, 2), NullWritable.get());
    } finally {
      Closeables.closeQuietly(writer);
    }

    EnumerateTrianglesJob enumerateTrianglesJob = new EnumerateTrianglesJob();
    enumerateTrianglesJob.setConf(conf);
    enumerateTrianglesJob.run(new String[] { "--input", inputFile.getAbsolutePath(),
        "--output", outputDir.getAbsolutePath(), "--tempDir", tempDir.getAbsolutePath() });

    Set<Triangle> triangles = Sets.newHashSet();
    for (Pair<Triangle,NullWritable> result :
        new SequenceFileIterable<Triangle, NullWritable>(new Path(outputDir.getAbsolutePath() + "/part-r-00000"),
        false, conf)) {
      triangles.add(result.getFirst());
    }

    assertEquals(6, triangles.size());
    assertTrue(triangles.contains(new Triangle(0, 1, 2)));
    assertTrue(triangles.contains(new Triangle(0, 1, 3)));
    assertTrue(triangles.contains(new Triangle(0, 2, 3)));
    assertTrue(triangles.contains(new Triangle(0, 4, 5)));
    assertTrue(triangles.contains(new Triangle(0, 4, 7)));
    assertTrue(triangles.contains(new Triangle(1, 2, 3)));
  }
}
