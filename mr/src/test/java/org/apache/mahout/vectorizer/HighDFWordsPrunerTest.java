package org.apache.mahout.vectorizer;
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

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakScope;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

@ThreadLeakScope(ThreadLeakScope.Scope.NONE)
public class HighDFWordsPrunerTest extends MahoutTestCase {
  private static final int NUM_DOCS = 100;

  private static final String[] HIGH_DF_WORDS = {"has", "which", "what", "srtyui"};

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

    for (int i = 0; i < NUM_DOCS; i++) {
      writer.append(new Text("Document::ID::" + i), new Text(enhanceWithHighDFWords(gen.getRandomDocument())));
    }
    writer.close();
  }

  private static String enhanceWithHighDFWords(String initialDoc) {
    StringBuilder sb = new StringBuilder(initialDoc);
    for (String word : HIGH_DF_WORDS) {
      sb.append(' ').append(word);
    }
    return sb.toString();
  }


  @Test
  public void testHighDFWordsPreserving() throws Exception {
    runTest(false);
  }

  @Test
  public void testHighDFWordsPruning() throws Exception {
    runTest(true);
  }

  private void runTest(boolean prune) throws Exception {
    Path outputPath = getTestTempFilePath("output");

    List<String> argList = Lists.newLinkedList();
    argList.add("-i");
    argList.add(inputPath.toString());
    argList.add("-o");
    argList.add(outputPath.toString());
    if (prune) {
      argList.add("-xs");
      argList.add("3"); // we prune all words that are outside 3*sigma
    } else {
      argList.add("--maxDFPercent");
      argList.add("100"); // the default if, -xs is not specified is to use maxDFPercent, which defaults to 99%
    }

    argList.add("-seq");
    argList.add("-nv");

    String[] args = argList.toArray(new String[argList.size()]);

    ToolRunner.run(conf, new SparseVectorsFromSequenceFiles(), args);

    Path dictionary = new Path(outputPath, "dictionary.file-0");
    Path tfVectors = new Path(outputPath, "tf-vectors");
    Path tfidfVectors = new Path(outputPath, "tfidf-vectors");

    int[] highDFWordsDictionaryIndices = getHighDFWordsDictionaryIndices(dictionary);
    validateVectors(tfVectors, highDFWordsDictionaryIndices, prune);
    validateVectors(tfidfVectors, highDFWordsDictionaryIndices, prune);
  }

  private int[] getHighDFWordsDictionaryIndices(Path dictionaryPath) {
    int[] highDFWordsDictionaryIndices = new int[HIGH_DF_WORDS.length];

    List<String> highDFWordsList = Arrays.asList(HIGH_DF_WORDS);

    for (Pair<Text, IntWritable> record : new SequenceFileDirIterable<Text, IntWritable>(dictionaryPath, PathType.GLOB,
            null, null, true, conf)) {
      int index = highDFWordsList.indexOf(record.getFirst().toString());
      if (index > -1) {
        highDFWordsDictionaryIndices[index] = record.getSecond().get();
      }
    }

    return highDFWordsDictionaryIndices;
  }

  private void validateVectors(Path vectorPath, int[] highDFWordsDictionaryIndices, boolean prune) throws Exception {
    assertTrue("Path does not exist", vectorPath.getFileSystem(conf).exists(vectorPath));
    for (VectorWritable value : new SequenceFileDirValueIterable<VectorWritable>(vectorPath, PathType.LIST, PathFilters
            .partFilter(), null, true, conf)) {
      Vector v = ((NamedVector) value.get()).getDelegate();
      for (int i = 0; i < highDFWordsDictionaryIndices.length; i++) {
        if (prune) {
          assertEquals("Found vector for which word '" + HIGH_DF_WORDS[i] + "' is not pruned", 0.0, v
              .get(highDFWordsDictionaryIndices[i]), 0.0);
        } else {
          assertTrue("Found vector for which word '" + HIGH_DF_WORDS[i] + "' is pruned, and shouldn't have been", v
                  .get(highDFWordsDictionaryIndices[i]) != 0.0);
        }
      }
    }
  }
}
