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

package org.apache.mahout.vectorizer;

import java.util.List;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterator;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

public class EncodedVectorsFromSequenceFilesTest extends MahoutTestCase {

  private static final int NUM_DOCS = 100;
  
  private Configuration conf;
  private Path inputPath;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    conf = getConfiguration();

    inputPath = getTestTempFilePath("documents/docs.file");
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, inputPath, Text.class, Text.class);

    RandomDocumentGenerator gen = new RandomDocumentGenerator();

    try {
      for (int i = 0; i < NUM_DOCS; i++) {
        writer.append(new Text("Document::ID::" + i), new Text(gen.getRandomDocument()));
      }
    } finally {
      Closeables.close(writer, false);
    }
  }
  
  
  @Test
  public void testCreate() throws Exception {
    runTest(false, false);
  }

  @Test
  public void testCreateNamed() throws Exception {
    runTest(false, true);
  }
  
  @Test
  public void testCreateSeq() throws Exception {
    runTest(true, false);
  }
  
  @Test
  public void testCreateSeqNamed() throws Exception {
    runTest(true, true);
  }
  
  private void runTest(boolean sequential, boolean named) throws Exception {
    Path tmpPath = getTestTempDirPath();
    Path outputPath = new Path(tmpPath, "output");
    
    List<String> argList = Lists.newLinkedList();;
    argList.add("-i");
    argList.add(inputPath.toString());
    argList.add("-o");
    argList.add(outputPath.toString());
    
    if (sequential) {
      argList.add("-seq");
    }
    
    if (named) {
      argList.add("-nv");
    }
    
    String[] args = argList.toArray(new String[argList.size()]);

    ToolRunner.run(getConfiguration(), new EncodedVectorsFromSequenceFiles(), args);

    SequenceFileDirIterator<Text, VectorWritable> iter = new SequenceFileDirIterator<Text, VectorWritable>(outputPath, PathType.LIST, PathFilters.partFilter(), null, true, conf);
    int seen = 0;
    while (iter.hasNext()) {
      Pair<Text, VectorWritable> next = iter.next();
      if (sequential && !named) {
        assertTrue(next.getSecond().get() instanceof SequentialAccessSparseVector);
      } else if (named) {
        assertTrue(next.getSecond().get() instanceof NamedVector);
      }
      seen++;
    }
    assertEquals("Missed some vectors", NUM_DOCS, seen);
  }  
}
