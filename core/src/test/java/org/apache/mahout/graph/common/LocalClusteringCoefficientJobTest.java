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

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.graph.model.Triangle;
import org.apache.mahout.graph.model.UndirectedEdge;
import org.junit.Test;

import java.io.File;
import java.util.Map;

public class LocalClusteringCoefficientJobTest extends MahoutTestCase {

  private static final Splitter TAB = Splitter.on('\t');

  @Test
  public void toyIntegrationTest() throws Exception {

    File edgesFile = getTestTempFile("edges.seq");
    File trianglesFile = getTestTempFile("triangles.seq");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tempDir = getTestTempDir("tmp");

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    SequenceFile.Writer edgesWriter = new SequenceFile.Writer(fs, conf, new Path(edgesFile.getAbsolutePath()),
        UndirectedEdge.class, NullWritable.class);
    try {
      edgesWriter.append(new UndirectedEdge(0, 1), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(0, 2), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(0, 3), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(0, 4), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(0, 5), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(0, 6), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(0, 7), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(1, 2), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(1, 3), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(2, 3), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(4, 5), NullWritable.get());
      edgesWriter.append(new UndirectedEdge(4, 7), NullWritable.get());
    } finally {
      Closeables.closeQuietly(edgesWriter);
    }

    SequenceFile.Writer trianglesWriter = new SequenceFile.Writer(fs, conf, new Path(trianglesFile.getAbsolutePath()),
        Triangle.class, NullWritable.class);
    try {
      trianglesWriter.append(new Triangle(0, 1, 2), NullWritable.get());
      trianglesWriter.append(new Triangle(0, 1, 3), NullWritable.get());
      trianglesWriter.append(new Triangle(0, 2, 3), NullWritable.get());
      trianglesWriter.append(new Triangle(0, 4, 5), NullWritable.get());
      trianglesWriter.append(new Triangle(0, 4, 7), NullWritable.get());
      trianglesWriter.append(new Triangle(1, 2, 3), NullWritable.get());
    } finally {
      Closeables.closeQuietly(trianglesWriter);
    }

    LocalClusteringCoefficientJob clusteringCoefficientJob = new LocalClusteringCoefficientJob();
    clusteringCoefficientJob.setConf(conf);
    clusteringCoefficientJob.run(new String[] { "--edges", edgesFile.getAbsolutePath(),
        "--triangles", trianglesFile.getAbsolutePath(), "--output", outputDir.getAbsolutePath(),
        "--tempDir", tempDir.getAbsolutePath() });

    Map<Long,Double> localClusteringCoefficients = Maps.newHashMap();
    for (CharSequence line : new FileLineIterable(new File(outputDir, "part-r-00000"))) {
      String[] tokens = Iterables.toArray(TAB.split(line), String.class);
      localClusteringCoefficients.put(Long.parseLong(tokens[0]), Double.parseDouble(tokens[1]));
    }

    assertEquals(8, localClusteringCoefficients.size());
    assertEquals(0.119047, localClusteringCoefficients.get(0L), EPSILON);
    assertEquals(0.5, localClusteringCoefficients.get(1L), EPSILON);
    assertEquals(0.5, localClusteringCoefficients.get(2L), EPSILON);
    assertEquals(0.5, localClusteringCoefficients.get(3L), EPSILON);
    assertEquals(0.333333, localClusteringCoefficients.get(4L), EPSILON);
    assertEquals(0.5, localClusteringCoefficients.get(5L), EPSILON);
    assertEquals(0.0, localClusteringCoefficients.get(6L), EPSILON);
    assertEquals(0.5, localClusteringCoefficients.get(7L), EPSILON);
  }
}
