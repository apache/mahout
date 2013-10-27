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

package org.apache.mahout.utils.vectors.io;

import java.io.StringWriter;
import java.util.Collection;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.RandomVectorIterable;
import org.junit.Test;

public final class VectorWriterTest extends MahoutTestCase {

  @Test
  public void testSFVW() throws Exception {
    Path path = getTestTempFilePath("sfvw");
    Configuration conf = getConfiguration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter = new SequenceFile.Writer(fs, conf, path, LongWritable.class, VectorWritable.class);
    SequenceFileVectorWriter writer = new SequenceFileVectorWriter(seqWriter);
    try {
      writer.write(new RandomVectorIterable(50));
    } finally {
      Closeables.close(writer, false);
    }

    long count = HadoopUtil.countRecords(path, conf);
    assertEquals(50, count);
  }

  @Test
  public void testTextOutputSize() throws Exception {
    StringWriter strWriter = new StringWriter();
    VectorWriter writer = new TextualVectorWriter(strWriter);
    try {
      Collection<Vector> vectors = Lists.newArrayList();
      vectors.add(new DenseVector(new double[]{0.3, 1.5, 4.5}));
      vectors.add(new DenseVector(new double[]{1.3, 1.5, 3.5}));
      writer.write(vectors);
    } finally {
      Closeables.close(writer, false);
    }
    String buffer = strWriter.toString();
    assertNotNull(buffer);
    assertFalse(buffer.isEmpty());
    
  }
}
