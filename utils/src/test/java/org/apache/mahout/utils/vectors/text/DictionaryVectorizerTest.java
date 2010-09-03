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

import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.text.DefaultAnalyzer;
import org.apache.mahout.utils.MahoutTestCase;
import org.apache.mahout.utils.vectors.tfidf.TFIDFConverter;
import org.junit.Before;
import org.junit.Test;

/**
 * Test the dictionary Vector
 */
public final class DictionaryVectorizerTest extends MahoutTestCase {

  private static final int AVG_DOCUMENT_LENGTH = 20;
  private static final int AVG_SENTENCE_LENGTH = 8;
  private static final int AVG_WORD_LENGTH = 6;
  private static final int NUM_DOCS = 100;
  private static final String CHARSET = "abcdef";
  private static final String DELIM = " .,?;:!\t\n\r";
  private static final String ERRORSET = "`1234567890" + "-=~@#$%^&*()_+[]{}'\"/<>|\\";
  private static final Random random = RandomUtils.getRandom();

  private FileSystem fs;

  private static char getRandomDelimiter() {
    return DELIM.charAt(random.nextInt(DictionaryVectorizerTest.DELIM.length()));
  }

  private static String getRandomDocument() {
    int length = (AVG_DOCUMENT_LENGTH >> 1) + DictionaryVectorizerTest.random.nextInt(AVG_DOCUMENT_LENGTH);
    StringBuilder sb = new StringBuilder(length * AVG_SENTENCE_LENGTH * AVG_WORD_LENGTH);
    for (int i = 0; i < length; i++) {
      sb.append(getRandomSentence());
    }
    return sb.toString();
  }

  private static String getRandomSentence() {
    int length = (AVG_SENTENCE_LENGTH >> 1) + DictionaryVectorizerTest.random.nextInt(AVG_SENTENCE_LENGTH);
    StringBuilder sb = new StringBuilder(length * AVG_WORD_LENGTH);
    for (int i = 0; i < length; i++) {
      sb.append(getRandomString()).append(' ');
    }
    sb.append(getRandomDelimiter());
    return sb.toString();
  }

  private static String getRandomString() {
    int length = (AVG_WORD_LENGTH >> 1) + DictionaryVectorizerTest.random.nextInt(AVG_WORD_LENGTH);
    StringBuilder sb = new StringBuilder(length);
    for (int i = 0; i < length; i++) {
      sb.append(DictionaryVectorizerTest.CHARSET.charAt(
          DictionaryVectorizerTest.random.nextInt(DictionaryVectorizerTest.CHARSET.length())));
    }
    if (DictionaryVectorizerTest.random.nextInt(10) == 0) {
      sb.append(DictionaryVectorizerTest.ERRORSET.charAt(
          DictionaryVectorizerTest.random.nextInt(DictionaryVectorizerTest.ERRORSET.length())));
    }
    return sb.toString();
  }

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    Configuration conf = new Configuration();
    fs = FileSystem.get(conf);
  }

  @Test
  public void testCreateTermFrequencyVectors() throws Exception {
    Configuration conf = new Configuration();
    Path path = getTestTempFilePath("documents/docs.file");
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path, Text.class, Text.class);

    for (int i = 0; i < NUM_DOCS; i++) {
      writer.append(new Text("Document::ID::" + i), new Text(getRandomDocument()));
    }
    writer.close();
    Class<? extends Analyzer> analyzer = DefaultAnalyzer.class;
    DocumentProcessor.tokenizeDocuments(path, analyzer, getTestTempDirPath("output/tokenized-documents"));
    DictionaryVectorizer.createTermFrequencyVectors(getTestTempDirPath("output/tokenized-documents"),
                                                    getTestTempDirPath("output/wordcount"),
                                                    conf,
                                                    2,
                                                    1,
                                                    0.0f,
                                                    1,
                                                    100,
                                                    false);
    TFIDFConverter.processTfIdf(getTestTempDirPath("output/wordcount/tf-vectors"),
                                getTestTempDirPath("output/tfidf"),
                                100,
                                1,
                                99,
                                1.0f,
                                false,
                                1);

  }
}
