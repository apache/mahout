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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;
import org.junit.Before;
import org.junit.Test;

/**
 * Test the dictionary Vector
 */
public final class DictionaryVectorizerTest extends MahoutTestCase {

  private static final int NUM_DOCS = 100;
  
  private Configuration conf;
  private FileSystem fs;
  private Path inputPath;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    conf = new Configuration();
    fs = FileSystem.get(conf);

    inputPath = getTestTempFilePath("documents/docs.file");
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, inputPath, Text.class, Text.class);

    RandomDocumentGenerator gen = new RandomDocumentGenerator();
    
    for (int i = 0; i < NUM_DOCS; i++) {
      writer.append(new Text("Document::ID::" + i), new Text(gen.getRandomDocument()));
    }
    writer.close();
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
  
  public void runTest(boolean sequential, boolean named) throws Exception {
    
    Class<? extends Analyzer> analyzer = DefaultAnalyzer.class;
    
    Path tokenizedDocuments = getTestTempDirPath("output/tokenized-documents");
    Path wordCount = getTestTempDirPath("output/wordcount");
    Path tfVectors = new Path(wordCount, "tf-vectors");
    Path tfidf = getTestTempDirPath("output/tfidf");
    Path tfidfVectors = new Path(tfidf, "tfidf-vectors");
    
    DocumentProcessor.tokenizeDocuments(inputPath, analyzer, tokenizedDocuments);
    
    DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocuments,
                                                    wordCount,
                                                    conf,
                                                    2,
                                                    1,
                                                    0.0f,
                                                    -1.0f,
                                                    true,
                                                    1,
                                                    100,
                                                    sequential,
                                                    named);
    
    validateVectors(fs, conf, NUM_DOCS, tfVectors, sequential, named);
    
    TFIDFConverter.processTfIdf(tfVectors,
                                tfidf,
                                100,
                                1,
                                99,
                                2.0f,
                                false,
                                sequential,
                                named,
                                1);
    
    
    validateVectors(fs, conf, NUM_DOCS, tfidfVectors, sequential, named);
  }
  
  public static void validateVectors(FileSystem fs,
                                     Configuration conf,
                                     int numDocs,
                                     Path vectorPath,
                                     boolean sequential,
                                     boolean named) throws Exception {
    FileStatus[] stats = fs.listStatus(vectorPath, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        return path.getName().startsWith("part-");
      }
      
    });

    int count = 0;
    Writable key = new Text();
    VectorWritable vw = new VectorWritable();
    for (FileStatus s: stats) {
      SequenceFile.Reader tfidfReader = new SequenceFile.Reader(fs, s.getPath(), conf);
      while (tfidfReader.next(key, vw)) {
        count++;
        Vector v = vw.get();
        if (named) {
          assertTrue("Expected NamedVector", v instanceof NamedVector);
          v = ((NamedVector) v).getDelegate();
        }
        
        if (sequential) {
          assertTrue("Expected SequentialAccessSparseVector", v instanceof SequentialAccessSparseVector);
        }
        else {
          assertTrue("Expected RandomAccessSparseVector", v instanceof RandomAccessSparseVector);
        }
        
      }
      tfidfReader.close();
    }

    assertEquals("Expected " + numDocs + " documents", numDocs, count);
  }
}
