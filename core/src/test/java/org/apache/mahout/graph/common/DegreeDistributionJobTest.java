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
import org.apache.mahout.graph.model.UndirectedEdge;
import org.junit.Test;

import java.io.File;
import java.util.Map;

public class DegreeDistributionJobTest extends MahoutTestCase {

  private static final Splitter TAB = Splitter.on('\t');

  @Test
  public void toyIntegrationTest() throws Exception {

    File inputFile = getTestTempFile("edges.seq");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tempDir = getTestTempDir("tmp");

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(inputFile.getAbsolutePath()),
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

    DegreeDistributionJob degreeDistributionJob = new DegreeDistributionJob();
    degreeDistributionJob.setConf(conf);
    degreeDistributionJob.run(new String[] { "--input", inputFile.getAbsolutePath(),
        "--output", outputDir.getAbsolutePath(), "--tempDir", tempDir.getAbsolutePath() });

    Map<Integer,Integer> degreeDistribution = Maps.newHashMap();
    for (CharSequence line : new FileLineIterable(new File(outputDir, "part-r-00000"))) {
      String[] tokens = Iterables.toArray(TAB.split(line), String.class);
      degreeDistribution.put(Integer.parseInt(tokens[0]), Integer.parseInt(tokens[1]));
    }

    assertEquals(4, degreeDistribution.size());
    assertEquals(1, degreeDistribution.get(1).intValue());
    assertEquals(2, degreeDistribution.get(2).intValue());
    assertEquals(4, degreeDistribution.get(3).intValue());
    assertEquals(1, degreeDistribution.get(7).intValue());
  }

}
