/*
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

package org.apache.mahout.clustering.streaming.tools;

import com.google.common.collect.Iterables;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.junit.Test;

public class ResplitSequenceFilesTest extends MahoutTestCase {

  @Test
  public void testSplitting() throws Exception {

    Path inputFile = new Path(getTestTempDirPath("input"), "test.seq");
    Path output = getTestTempDirPath("output");
    Configuration conf = new Configuration();
    LocalFileSystem fs = FileSystem.getLocal(conf);

    SequenceFile.Writer writer = null;
    try {
      writer = SequenceFile.createWriter(fs, conf, inputFile, IntWritable.class, IntWritable.class);
      writer.append(new IntWritable(1), new IntWritable(1));
      writer.append(new IntWritable(2), new IntWritable(2));
      writer.append(new IntWritable(3), new IntWritable(3));
      writer.append(new IntWritable(4), new IntWritable(4));
      writer.append(new IntWritable(5), new IntWritable(5));
      writer.append(new IntWritable(6), new IntWritable(6));
      writer.append(new IntWritable(7), new IntWritable(7));
      writer.append(new IntWritable(8), new IntWritable(8));
    } finally {
      Closeables.close(writer, false);
    }

    String splitPattern = "split";
    int numSplits = 4;

    ResplitSequenceFiles.main(new String[] { "--input", inputFile.toString(),
        "--output", output.toString() + "/" + splitPattern, "--numSplits", String.valueOf(numSplits) });

    FileStatus[] statuses = HadoopUtil.getFileStatus(output, PathType.LIST, PathFilters.logsCRCFilter(), null, conf);

    for (FileStatus status : statuses) {
      String name = status.getPath().getName();
      assertTrue(name.startsWith(splitPattern));
      assertEquals(2, numEntries(status, conf));
    }
    assertEquals(numSplits, statuses.length);
  }

  private int numEntries(FileStatus status, Configuration conf) {
    return Iterables.size(new SequenceFileIterable(status.getPath(), conf));
  }
}
