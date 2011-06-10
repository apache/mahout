/*
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

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.ClassifierData;
import org.apache.mahout.examples.MahoutTestCase;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.junit.Before;
import org.junit.Test;

public final class SplitBayesInputTest extends MahoutTestCase {

  private OpenObjectIntHashMap<String> countMap;
  private Charset charset;
  private FileSystem fs;
  private Path tempInputFile;
  private Path tempTrainingDirectory;
  private Path tempTestDirectory;
  private Path tempInputDirectory;
  private SplitBayesInput si;
    
  @Override
  @Before
  public void setUp() throws Exception {
    Configuration conf = new Configuration();
    fs   = FileSystem.get(conf);
    
    super.setUp();
  
    countMap = new OpenObjectIntHashMap<String>();
    
    charset = Charsets.UTF_8;
    tempInputFile = getTestTempFilePath("bayesinputfile");
    tempTrainingDirectory = getTestTempDirPath("bayestrain");
    tempTestDirectory = getTestTempDirPath("bayestest");
    tempInputDirectory = getTestTempDirPath("bayesinputdir");
    
    si = new SplitBayesInput();
    si.setTrainingOutputDirectory(tempTrainingDirectory);
    si.setTestOutputDirectory(tempTestDirectory);
    si.setInputDirectory(tempInputDirectory);
  }
  
  private void writeMultipleInputFiles() throws IOException {
    Writer writer = null;
    String currentLabel = null;
    
    for (String[] entry : ClassifierData.DATA) {
      if (!entry[0].equals(currentLabel)) {
        currentLabel = entry[0];
        Closeables.closeQuietly(writer);
        
        writer = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(tempInputDirectory, currentLabel)),
            Charsets.UTF_8));
      }
      countMap.adjustOrPutValue(currentLabel, 1, 1);
      writer.write(currentLabel + '\t' + entry[1] + '\n');
    }
    Closeables.closeQuietly(writer);
  }

  private void writeSingleInputFile() throws IOException {
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.create(tempInputFile), Charsets.UTF_8));
    try {
      for (String[] entry : ClassifierData.DATA) {
        writer.write(entry[0] + '\t' + entry[1] + '\n');
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }

  @Test
  public void testSplitDirectory() throws Exception {

    writeMultipleInputFiles();

    final int testSplitSize = 1;
    si.setTestSplitSize(testSplitSize);
    si.setCallback(new SplitBayesInput.SplitCallback() {
          @Override
          public void splitComplete(Path inputFile, int lineCount, int trainCount, int testCount, int testSplitStart) {
            int trainingLines = countMap.get(inputFile.getName()) - testSplitSize;
            assertSplit(fs, inputFile, charset, testSplitSize, trainingLines, tempTrainingDirectory, tempTestDirectory);
          }
    });
    
    si.splitDirectory(tempInputDirectory);
  }

  @Test
  public void testSplitFile() throws Exception {
    writeSingleInputFile();
    si.setTestSplitSize(2);
    si.setCallback(new TestCallback(2, 10));
    si.splitFile(tempInputFile);
  }

  @Test
  public void testSplitFileLocation() throws Exception {
    writeSingleInputFile();
    si.setTestSplitSize(2);
    si.setSplitLocation(50);
    si.setCallback(new TestCallback(2, 10));
    si.splitFile(tempInputFile);
  }

  @Test
  public void testSplitFilePct() throws Exception {
    writeSingleInputFile();
    si.setTestSplitPct(25);
   
    si.setCallback(new TestCallback(3, 9));
    si.splitFile(tempInputFile);
  }

  @Test
  public void testSplitFilePctLocation() throws Exception {
    writeSingleInputFile();
    si.setTestSplitPct(25);
    si.setSplitLocation(50);
    si.setCallback(new TestCallback(3, 9));
    si.splitFile(tempInputFile);
  }

  @Test
  public void testSplitFileRandomSelectionSize() throws Exception {
    writeSingleInputFile();
    si.setTestRandomSelectionSize(5);
   
    si.setCallback(new TestCallback(5, 7));
    si.splitFile(tempInputFile);
  }

  @Test
  public void testSplitFileRandomSelectionPct() throws Exception {
    writeSingleInputFile();
    si.setTestRandomSelectionPct(25);
   
    si.setCallback(new TestCallback(3, 9));
    si.splitFile(tempInputFile);
  }

  @Test
  public void testValidate() throws Exception {
    SplitBayesInput st = new SplitBayesInput();
    assertValidateException(st);
    
    st.setTestSplitSize(100);
    assertValidateException(st);
    
    st.setTestOutputDirectory(tempTestDirectory);
    assertValidateException(st);
    
    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();
    
    st.setTestSplitPct(50);
    assertValidateException(st);
    
    st = new SplitBayesInput();
    st.setTestRandomSelectionPct(50);
    st.setTestOutputDirectory(tempTestDirectory);
    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();
    
    st.setTestSplitPct(50);
    assertValidateException(st);
    
    st = new SplitBayesInput();
    st.setTestRandomSelectionPct(50);
    st.setTestOutputDirectory(tempTestDirectory);
    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();
    
    st.setTestSplitSize(100);
    assertValidateException(st);
  }
  
  private class TestCallback implements SplitBayesInput.SplitCallback {
    private final int testSplitSize;
    private final int trainingLines;
    
    private TestCallback(int testSplitSize, int trainingLines) {
      this.testSplitSize = testSplitSize;
      this.trainingLines = trainingLines;
    }
    
    @Override
    public void splitComplete(Path inputFile, int lineCount, int trainCount, int testCount, int testSplitStart) {
      assertSplit(fs, tempInputFile, charset, testSplitSize, trainingLines, tempTrainingDirectory, tempTestDirectory);
    }
  }

  private static void assertValidateException(SplitBayesInput st) throws IOException {
    try {
      st.validate();
      fail("Expected IllegalArgumentException");
    } catch (IllegalArgumentException iae) {
      // good
    }
  }
  
  private static void assertSplit(FileSystem fs,
                                  Path tempInputFile,
                                  Charset charset,
                                  int testSplitSize,
                                  int trainingLines,
                                  Path tempTrainingDirectory,
                                  Path tempTestDirectory) {

    try {
      Path testFile = new Path(tempTestDirectory, tempInputFile.getName());
      //assertTrue("test file exists", testFile.isFile());
      assertEquals("test line count", testSplitSize, SplitBayesInput.countLines(fs, testFile, charset));

      Path trainingFile = new Path(tempTrainingDirectory, tempInputFile.getName());
      //assertTrue("training file exists", trainingFile.isFile());
      assertEquals("training line count", trainingLines, SplitBayesInput.countLines(fs, trainingFile, charset));
    } catch (IOException ioe) {
      fail(ioe.toString());
    }
  }
}
