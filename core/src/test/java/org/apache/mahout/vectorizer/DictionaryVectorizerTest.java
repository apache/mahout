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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakLingering;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
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
@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public final class DictionaryVectorizerTest extends MahoutTestCase {

  private static final int NUM_DOCS = 100;

  private Path inputPath;
  
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = getConfiguration();

    inputPath = getTestTempFilePath("documents/docs.file");
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, inputPath, Text.class, Text.class);

    try {
      RandomDocumentGenerator gen = new RandomDocumentGenerator();

      for (int i = 0; i < NUM_DOCS; i++) {
        writer.append(new Text("Document::ID::" + i), new Text(gen.getRandomDocument()));
      }
    } finally {
      Closeables.close(writer, false);
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
  
  private void runTest(boolean sequential, boolean named)
    throws IOException, ClassNotFoundException, InterruptedException {
    
    Class<? extends Analyzer> analyzer = StandardAnalyzer.class;
    
    Path tokenizedDocuments = getTestTempDirPath("output/tokenized-documents");
    Path wordCount = getTestTempDirPath("output/wordcount");
    Path tfVectors = new Path(wordCount, "tf-vectors");
    Path tfidf = getTestTempDirPath("output/tfidf");
    Path tfidfVectors = new Path(tfidf, "tfidf-vectors");
    
    Configuration conf = getConfiguration();
    DocumentProcessor.tokenizeDocuments(inputPath, analyzer, tokenizedDocuments, conf);
    
    DictionaryVectorizer.createTermFrequencyVectors(tokenizedDocuments,
                                                    wordCount,
                                                    DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER,
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
    
    validateVectors(conf, NUM_DOCS, tfVectors, sequential, named);
    
    Pair<Long[], List<Path>> docFrequenciesFeatures = TFIDFConverter.calculateDF(tfVectors, 
    		tfidf, conf, 100);

    TFIDFConverter.processTfIdf(tfVectors,
                                tfidf,
                                conf,
                                docFrequenciesFeatures,
                                1,
                                -1,
                                2.0f,
                                false,
                                sequential,
                                named,
                                1);
    
    
    validateVectors(conf, NUM_DOCS, tfidfVectors, sequential, named);
  }
  
  public static void validateVectors(Configuration conf,
                                     int numDocs,
                                     Path vectorPath,
                                     boolean sequential,
                                     boolean named) {
    int count = 0;
    for (VectorWritable value :
         new SequenceFileDirValueIterable<VectorWritable>(
             vectorPath, PathType.LIST, PathFilters.partFilter(), null, true, conf)) {
      count++;
      Vector v = value.get();
      if (named) {
        assertTrue("Expected NamedVector", v instanceof NamedVector);
        v = ((NamedVector) v).getDelegate();
      }

      if (sequential) {
        assertTrue("Expected SequentialAccessSparseVector", v instanceof SequentialAccessSparseVector);
      } else {
        assertTrue("Expected RandomAccessSparseVector", v instanceof RandomAccessSparseVector);
      }

    }

  assertEquals("Expected " + numDocs + " documents", numDocs, count);
  }
}
