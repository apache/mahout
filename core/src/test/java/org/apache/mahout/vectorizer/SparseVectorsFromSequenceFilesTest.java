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

import java.io.IOException;
import java.util.List;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Test;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class SparseVectorsFromSequenceFilesTest extends MahoutTestCase {

  private static final int NUM_DOCS = 100;
  
  private Configuration conf;
  private Path inputPath;

  private void setupDocs() throws IOException {
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
  public void testCreateTermFrequencyVectors() throws Exception {
    setupDocs();
    runTest(false, false, false, -1, NUM_DOCS);
  }

  @Test
  public void testCreateTermFrequencyVectorsNam() throws Exception {
    setupDocs();
    runTest(false, false, true, -1, NUM_DOCS);
  }
  
  @Test
  public void testCreateTermFrequencyVectorsSeq() throws Exception {
    setupDocs();
    runTest(false, true, false, -1, NUM_DOCS);
  }
  
  @Test
  public void testCreateTermFrequencyVectorsSeqNam() throws Exception {
    setupDocs();
    runTest(false, true, true, -1, NUM_DOCS);
  }

  @Test
  public void testPruning() throws Exception {
    conf = getConfiguration();
    inputPath = getTestTempFilePath("documents/docs.file");
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, inputPath, Text.class, Text.class);

    String [] docs = {"a b c", "a a a a a b", "a a a a a c"};

    try {
      for (int i = 0; i < docs.length; i++) {
        writer.append(new Text("Document::ID::" + i), new Text(docs[i]));
      }
    } finally {
      Closeables.close(writer, false);
    }
    Path outPath = runTest(false, false, false, 2, docs.length);
    Path tfidfVectors = new Path(outPath, "tfidf-vectors");
    int count = 0;
    Vector [] res = new Vector[docs.length];
    for (VectorWritable value :
         new SequenceFileDirValueIterable<VectorWritable>(
             tfidfVectors, PathType.LIST, PathFilters.partFilter(), null, true, conf)) {
      Vector v = value.get();
      System.out.println(v);
      assertEquals(2, v.size());
      res[count] = v;
      count++;
    }
    assertEquals(docs.length, count);
    //the first doc should have two values, the second and third should have 1, since the a gets removed
    assertEquals(2, res[0].getNumNondefaultElements());
    assertEquals(1, res[1].getNumNondefaultElements());
    assertEquals(1, res[2].getNumNondefaultElements());
  }

  @Test
  public void testPruningTF() throws Exception {
    conf = getConfiguration();
    FileSystem fs = FileSystem.get(conf);

    inputPath = getTestTempFilePath("documents/docs.file");
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, inputPath, Text.class, Text.class);

    String [] docs = {"a b c", "a a a a a b", "a a a a a c"};

    try {
      for (int i = 0; i < docs.length; i++) {
        writer.append(new Text("Document::ID::" + i), new Text(docs[i]));
      }
    } finally {
      Closeables.close(writer, false);
    }
    Path outPath = runTest(true, false, false, 2, docs.length);
    Path tfVectors = new Path(outPath, "tf-vectors");
    int count = 0;
    Vector [] res = new Vector[docs.length];
    for (VectorWritable value :
         new SequenceFileDirValueIterable<VectorWritable>(
             tfVectors, PathType.LIST, PathFilters.partFilter(), null, true, conf)) {
      Vector v = value.get();
      System.out.println(v);
      assertEquals(2, v.size());
      res[count] = v;
      count++;
    }
    assertEquals(docs.length, count);
    //the first doc should have two values, the second and third should have 1, since the a gets removed
    assertEquals(2, res[0].getNumNondefaultElements());
    assertEquals(1, res[1].getNumNondefaultElements());
    assertEquals(1, res[2].getNumNondefaultElements());
  }

  private Path runTest(boolean tfWeighting, boolean sequential, boolean named, double maxDFSigma, int numDocs) throws Exception {
    Path outputPath = getTestTempFilePath("output");

    List<String> argList = Lists.newLinkedList();
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
    if (maxDFSigma >= 0) {
      argList.add("--maxDFSigma");
      argList.add(String.valueOf(maxDFSigma));
    }
    if (tfWeighting) {
      argList.add("--weight");
      argList.add("tf");
    }
    String[] args = argList.toArray(new String[argList.size()]);
    
    ToolRunner.run(getConfiguration(), new SparseVectorsFromSequenceFiles(), args);

    Path tfVectors = new Path(outputPath, "tf-vectors");
    Path tfidfVectors = new Path(outputPath, "tfidf-vectors");
    
    DictionaryVectorizerTest.validateVectors(conf, numDocs, tfVectors, sequential, named);
    if (!tfWeighting) {
      DictionaryVectorizerTest.validateVectors(conf, numDocs, tfidfVectors, sequential, named);
    }
    return outputPath;
  }  
}
