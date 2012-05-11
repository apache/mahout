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

package org.apache.mahout.math.stats.entropy;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.junit.Test;

import java.util.Iterator;

public final class ConditionalEntropyTest extends MahoutTestCase {

  @Test
  public void testConditionalEntropy() throws Exception {

    Configuration configuration = new Configuration();
    Path input = getTestTempFilePath("input");
    Path output = getTestTempFilePath("output");
    FileSystem fileSystem = FileSystem.get(input.toUri(), configuration);

    // create input
    String[] keys = { "Math", "History", "CS", "Math", "Math", "CS", "History", "Math" };
    String[] values = { "Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes" };
    SequenceFile.Writer writer = new SequenceFile.Writer(fileSystem, configuration, input, Text.class, Text.class);
    try {
      for (int i = 0; i < keys.length; i++) {
        writer.append(new Text(keys[i]), new Text(values[i]));
      }
    } finally {
      Closeables.closeQuietly(writer);
    }

    // run the job
    Tool job = new ConditionalEntropy();
    String[] args = { "-i", input.toString(), "-o", output.toString(),
        "--tempDir", getTestTempDirPath("tmp").toString() };
    ToolRunner.run(job, args);

    // check the output
    Iterator<DoubleWritable> iteratorNodes =
        new SequenceFileDirValueIterator<DoubleWritable>(output,
                                                         PathType.LIST,
                                                         PathFilters.logsCRCFilter(),
                                                         null,
                                                         false,
                                                         new Configuration());
    while (iteratorNodes.hasNext()) {
      assertEquals(0.5, iteratorNodes.next().get(), EPSILON);
    }

  }

}
