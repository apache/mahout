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
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.graph.model.UndirectedEdge;
import org.apache.mahout.graph.model.Vertex;
import org.easymock.EasyMock;
import org.junit.Test;

public class SimplifyGraphJobTest extends MahoutTestCase {

  @Test
  public void testEdgeMapping() throws Exception {
    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);

    ctx.write(new UndirectedEdge(new Vertex(123L), new Vertex(456L)), NullWritable.get());

    EasyMock.replay(ctx);

    new SimplifyGraphJob.SimplifyGraphMapper().map(null, new Text("123,456"), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void testLoopRemoval() throws Exception {
    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);

    EasyMock.replay(ctx);

    new SimplifyGraphJob.SimplifyGraphMapper().map(null, new Text("123,123"), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void testIgnoreUnparseableLines() throws Exception {
    Mapper.Context ctx = EasyMock.createMock(Mapper.Context.class);

    EasyMock.replay(ctx);

    new SimplifyGraphJob.SimplifyGraphMapper().map(null, new Text("abc,123"), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void testAggregation() throws Exception {
    Reducer.Context ctx = EasyMock.createMock(Reducer.Context.class);
    UndirectedEdge edge = new UndirectedEdge(new Vertex(123L), new Vertex(456L));

    ctx.write(edge, NullWritable.get());

    EasyMock.replay(ctx);

    new SimplifyGraphJob.SimplifyGraphReducer().reduce(edge, Arrays.asList(NullWritable.get(), NullWritable.get()), ctx);

    EasyMock.verify(ctx);
  }

  @Test
  public void toyIntegrationTest() throws Exception {
    File inputFile = getTestTempFile("graph.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tempDir = getTestTempDir("tmp");

    writeLines(inputFile,
        "0,0",
        "0,1",
        "1,0",
        "1,0",
        "2,3",
        "4,3",
        "4,2");

    Configuration conf = new Configuration();
    SimplifyGraphJob simplifyGraphJob = new SimplifyGraphJob();
    simplifyGraphJob.setConf(conf);
    simplifyGraphJob.run(new String[] { "--input", inputFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--tempDir", tempDir.getAbsolutePath() });

    Set<UndirectedEdge> edges = Sets.newHashSet();
    for (Pair<UndirectedEdge,NullWritable> result :
        new SequenceFileIterable<UndirectedEdge, NullWritable>(new Path(outputDir.getAbsolutePath() + "/part-r-00000"),
        false, conf)) {
      edges.add(result.getFirst());
    }

    assertEquals(4, edges.size());
    assertTrue(edges.contains(new UndirectedEdge(1, 0)));
    assertTrue(edges.contains(new UndirectedEdge(2, 3)));
    assertTrue(edges.contains(new UndirectedEdge(2, 4)));
    assertTrue(edges.contains(new UndirectedEdge(3, 4)));
  }

}
