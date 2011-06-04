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

package org.apache.mahout.classifier.bayes;

import java.io.File;
import java.io.Writer;
import java.util.Iterator;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.mahout.classifier.BayesFileFormatter;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.FileLineIterator;
import org.apache.mahout.common.MahoutTestCase;
import org.junit.Before;
import org.junit.Test;

public final class BayesFileFormatterTest extends MahoutTestCase {

  private static final String[] WORDS = {"dog", "cat", "fish", "snake", "zebra"};

  private File input;
  private File out;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    input = getTestTempDir("bayes/in");
    out = getTestTempDir("bayes/out");
    for (String word : WORDS) {
      File file = new File(input, word);
      Writer writer = Files.newWriter(file, Charsets.UTF_8);
      try {
      writer.write(word);
      } finally {
        Closeables.closeQuietly(writer);
      }
    }
  }

  @Test
  public void test() throws Exception {
    Analyzer analyzer = new WhitespaceAnalyzer();
    File[] files = out.listFiles();
    assertEquals("files Size: " + files.length + " is not: " + 0, 0, files.length);
    BayesFileFormatter.format("animal", analyzer, input, Charsets.UTF_8, out);

    files = out.listFiles();
    assertEquals("files Size: " + files.length + " is not: " + WORDS.length, files.length, WORDS.length);
    for (File file : files) {
      //should only be one line in the file, and it should be label label
      Iterator<String> it = new FileLineIterator(file);
      String line = it.next().trim();
      assertFalse(it.hasNext());
      String label = "animal" + '\t' + file.getName();
      assertEquals(line + ":::: is not equal to " + label + "::::", line, label);
    }
  }

  @Test
  public void testCollapse() throws Exception {
    Analyzer analyzer = new WhitespaceAnalyzer();
    File[] files = out.listFiles();
    assertEquals("files Size: " + files.length + " is not: " + 0, 0, files.length);
    BayesFileFormatter.collapse("animal", analyzer, input, Charsets.UTF_8, new File(out, "animal"));
    files = out.listFiles();
    assertEquals("files Size: " + files.length + " is not: " + 1, 1, files.length);
    int count = 0;
    for (String line : new FileLineIterable(files[0])) {
      assertTrue("line does not start with label", line.startsWith("animal"));
      count++;
    }
    assertEquals(count + " does not equal: " + WORDS.length, count, WORDS.length);
  }

}