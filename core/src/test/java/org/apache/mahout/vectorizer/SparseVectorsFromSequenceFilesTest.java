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

import java.util.LinkedList;
import java.util.List;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Before;
import org.junit.Test;

public class SparseVectorsFromSequenceFilesTest extends MahoutTestCase {

  private static final int NUM_DOCS = 100;
  
  private Configuration conf;
  private Path inputPath;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    inputPath = getTestTempFilePath("documents/docs.file");
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, inputPath, Text.class, Text.class);

    RandomDocumentGenerator gen = new RandomDocumentGenerator();

    try {
      for (int i = 0; i < NUM_DOCS; i++) {
        writer.append(new Text("Document::ID::" + i), new Text(gen.getRandomDocument()));
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }
  
  
  @Test
  public void testCreateTermFrequencyVectors() throws Exception {
    runTest(false, false);
  }

  @Test
  public void testCreateTermFrequencyVectorsNam() throws Exception {
    runTest(false, true);
  }
  
  @Test
  public void testCreateTermFrequencyVectorsSeq() throws Exception {
    runTest(true, false);
  }
  
  @Test
  public void testCreateTermFrequencyVectorsSeqNam() throws Exception {
    runTest(true, true);
  }
  
  private void runTest(boolean sequential, boolean named) throws Exception {
    Path outputPath = getTestTempFilePath("output");

    
    List<String> argList = new LinkedList<String>();
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
    
    SparseVectorsFromSequenceFiles.main(args);

    Path tfVectors = new Path(outputPath, "tf-vectors");
    Path tfidfVectors = new Path(outputPath, "tfidf-vectors");
    
    DictionaryVectorizerTest.validateVectors(conf, NUM_DOCS, tfVectors, sequential, named);
    DictionaryVectorizerTest.validateVectors(conf, NUM_DOCS, tfidfVectors, sequential, named);
  }  
}
