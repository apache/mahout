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

package org.apache.mahout.utils.vectors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.io.SequenceFileVectorWriter;

import java.io.File;

public class SequenceFileVectorIterableTest extends MahoutTestCase {

  private File tmpLoc;
  private File tmpFile;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    File tmpDir = new File(System.getProperty("java.io.tmpdir"));
    tmpLoc = new File(tmpDir, "sfvit");
    tmpLoc.deleteOnExit();
    tmpLoc.mkdirs();
    tmpFile = File.createTempFile("sfvit", ".dat", tmpLoc);
    tmpFile.deleteOnExit();
  }

  @Override
  public void tearDown() throws Exception {
    tmpFile.delete();
    tmpLoc.delete();
    super.tearDown();
  }

  public void testIterable() throws Exception {
    Path path = new Path(tmpFile.getAbsolutePath());
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter = new SequenceFile.Writer(fs, conf, path, LongWritable.class, VectorWritable.class);
    SequenceFileVectorWriter writer = new SequenceFileVectorWriter(seqWriter);
    RandomVectorIterable iter = new RandomVectorIterable(50);
    writer.write(iter);
    writer.close();

    SequenceFile.Reader seqReader = new SequenceFile.Reader(fs, path, conf);
    SequenceFileVectorIterable sfvi = new SequenceFileVectorIterable(seqReader);
    int count = 0;
    for (Vector vector : sfvi) {
      //System.out.println("Vec: " + vector.asFormatString());
      count++;
    }
    seqReader.close();
    assertEquals(50, count);
  }
}
