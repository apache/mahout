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

import junit.framework.TestCase;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.WhitespaceAnalyzer;
import org.apache.mahout.classifier.BayesFileFormatter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;

public class BayesFileFormatterTest extends TestCase {

  protected File input;
  protected File out;
  protected String[] words;

  public BayesFileFormatterTest(String s) {
    super(s);
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    File tmpDir = new File(System.getProperty("java.io.tmpdir"));
    input = new File(tmpDir, "bayes/input");
    out = new File(tmpDir, "bayes/out");
    input.mkdirs();
    out.mkdirs();
    File[] files = out.listFiles();
    for (File file : files) {
      file.delete();
    }
    words = new String[]{"dog", "cat", "fish", "snake", "zebra"};
    for (String word : words) {
      File file = new File(input, word);
      Writer writer = new OutputStreamWriter(new FileOutputStream(file), Charset.forName("UTF-8"));
      writer.write(word);
      writer.close();
    }
  }

  public void test() throws IOException {
    Analyzer analyzer = new WhitespaceAnalyzer();
    File[] files = out.listFiles();
    assertEquals("files Size: " + files.length + " is not: " + 0, 0, files.length);
    Charset charset = Charset.forName("UTF-8");
    BayesFileFormatter.format("animal", analyzer, input, charset, out);

    files = out.listFiles();
    assertEquals("files Size: " + files.length + " is not: " + words.length, files.length, words.length);
    for (File file : files) {
      //should only be one line in the file, and it should be label label
      BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), charset));
      String line = reader.readLine();
      if (line != null) {
        line = line.trim();
      }
      String label = "animal" + '\t' + file.getName();
      assertEquals(line + ":::: is not equal to " + label + "::::", line, label);
      reader.close();
    }
  }

  public void testCollapse() throws Exception {
    Analyzer analyzer = new WhitespaceAnalyzer();
    File[] files = out.listFiles();
    assertEquals("files Size: " + files.length + " is not: " + 0, 0, files.length);
    Charset charset = Charset.forName("UTF-8");
    BayesFileFormatter.collapse("animal", analyzer, input, charset, new File(out, "animal"));
    files = out.listFiles();
    assertEquals("files Size: " + files.length + " is not: " + 1, 1, files.length);
    BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(files[0]), charset));
    String line;
    int count = 0;
    while ((line = reader.readLine()) != null) {
      assertTrue("line does not start with label", line.startsWith("animal"));
      System.out.println("Line: " + line);
      count++;
    }
    reader.close();
    assertEquals(count + " does not equal: " + words.length, count, words.length);

  }
}