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

package org.apache.mahout.utils.vectors.text;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.utils.vectors.tfidf.TFIDFConverter;

/**
 * Test the dictionary Vector
 */
public class DictionaryVectorizerTest extends MahoutTestCase {
  
  public static final int AVG_DOCUMENT_LENGTH = 20;
  
  public static final int AVG_SENTENCE_LENGTH = 8;
  
  public static final int AVG_WORD_LENGTH = 6;
  
  public static final int NUM_DOCS = 100;
  
  public static final String CHARSET = "abcdef";
  
  public static final String DELIM = " .,?;:!\t\n\r";
  
  public static final String ERRORSET = "`1234567890"
                                        + "-=~@#$%^&*()_+[]{}'\"/<>|\\";
  
  private static final Random random = RandomUtils.getRandom();
  
  private FileSystem fs;
  
  private static char getRandomDelimiter() {
    return DELIM.charAt(random.nextInt(DELIM.length()));
  }
  
  public static String getRandomDocument() {
    int length = (AVG_DOCUMENT_LENGTH >> 1)
                 + random.nextInt(AVG_DOCUMENT_LENGTH);
    StringBuilder sb = new StringBuilder(length * AVG_SENTENCE_LENGTH
                                         * AVG_WORD_LENGTH);
    for (int i = 0; i < length; i++) {
      sb.append(getRandomSentence());
    }
    return sb.toString();
  }
  
  public static String getRandomSentence() {
    int length = (AVG_SENTENCE_LENGTH >> 1)
                 + random.nextInt(AVG_SENTENCE_LENGTH);
    StringBuilder sb = new StringBuilder(length * AVG_WORD_LENGTH);
    for (int i = 0; i < length; i++) {
      sb.append(getRandomString()).append(' ');
    }
    sb.append(getRandomDelimiter());
    return sb.toString();
  }
  
  public static String getRandomString() {
    int length = (AVG_WORD_LENGTH >> 1) + random.nextInt(AVG_WORD_LENGTH);
    StringBuilder sb = new StringBuilder(length);
    for (int i = 0; i < length; i++) {
      sb.append(CHARSET.charAt(random.nextInt(CHARSET.length())));
    }
    if (random.nextInt(10) == 0) sb.append(ERRORSET.charAt(random
        .nextInt(ERRORSET.length())));
    return sb.toString();
  }
  
  private static void rmr(String path) throws Exception {
    File f = new File(path);
    if (f.exists()) {
      if (f.isDirectory()) {
        String[] contents = f.list();
        for (String content : contents) {
          rmr(f.toString() + File.separator + content);
        }
      }
      f.delete();
    }
  }
  
  @Override
  public void setUp() throws Exception {
    super.setUp();
    rmr("output");
    rmr("testdata");
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
  }
  
  public void testCreateTermFrequencyVectors() throws IOException,
                                              InterruptedException,
                                              ClassNotFoundException,
                                              URISyntaxException {
    Configuration conf = new Configuration();
    String pathString = "testdata/documents/docs.file";
    Path path = new Path(pathString);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path,
        Text.class, Text.class);
    
    for (int i = 0; i < NUM_DOCS; i++) {
      writer.append(new Text("Document::ID::" + i), new Text(
          getRandomDocument()));
    }
    writer.close();
    Class<? extends Analyzer> analyzer = new StandardAnalyzer(
        Version.LUCENE_CURRENT).getClass();
    DocumentProcessor.tokenizeDocuments(pathString, analyzer,
      "output/tokenized-documents");
    DictionaryVectorizer.createTermFrequencyVectors("output/tokenized-documents",
      "output/wordcount", 2, 1, 0.0f, 1, 100);
    TFIDFConverter.processTfIdf("output/wordcount/vectors", "output/tfidf/", 100, 1, 99, 1.0f);
    
  }
}
