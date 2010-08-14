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
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;

import junit.framework.TestCase;

import org.apache.mahout.classifier.ClassifierData;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.map.OpenObjectIntHashMap;


public class SplitBayesInputTest extends MahoutTestCase {

  OpenObjectIntHashMap<String> countMap;
  
  Charset charset;
  File tempInputFile;
  File tempTrainingDirectory;
  File tempTestDirectory;
  File tempInputDirectory;
  
  SplitBayesInput si;
    
  public void setUp() throws Exception {
    super.setUp();
  
    countMap = new OpenObjectIntHashMap<String>();
    
    charset = Charset.forName("UTF-8");
    tempInputFile = getTestTempFile("bayesinputfile");
    tempTrainingDirectory = getTestTempDir("bayestrain");
    tempTestDirectory = getTestTempDir("bayestest");
    tempInputDirectory = getTestTempDir("bayesinputdir");
    
    si = new SplitBayesInput();
    si.setTrainingOutputDirectory(tempTrainingDirectory);
    si.setTestOutputDirectory(tempTestDirectory);
    si.setInputDirectory(tempInputDirectory);
  }
  
  public void writeMultipleInputFiles() throws IOException {
    Writer writer = null;
    String currentLabel = null;
    
    for (String[] entry : ClassifierData.DATA) {
      if (!entry[0].equals(currentLabel)) {
        currentLabel = entry[0];
        if (writer != null) IOUtils.quietClose(writer);
        writer = new BufferedWriter(
            new OutputStreamWriter(
                new FileOutputStream(new File(tempInputDirectory, currentLabel)), Charset.forName("UTF-8")));
      }
      countMap.adjustOrPutValue(currentLabel, 1, 1);
      writer.write(currentLabel + '\t' + entry[1] + '\n');
    }
    IOUtils.quietClose(writer);
  }

  public void writeSingleInputFile() throws IOException {
    BufferedWriter writer = new BufferedWriter(
        new OutputStreamWriter(new FileOutputStream(tempInputFile), Charset.forName("UTF-8")));
    for (String[] entry : ClassifierData.DATA) {
      writer.write(entry[0] + '\t' + entry[1] + '\n');
    }
    writer.close();
  }
  
  public void testSplitDirectory() throws Exception {
    final int testSplitSize = 1;
    
    writeMultipleInputFiles();
    
    si.setTestSplitSize(testSplitSize);
    si.setCallback(new SplitBayesInput.SplitCallback() {
          @Override
          public void splitComplete(File inputFile, int lineCount, int trainCount, int testCount, int testSplitStart) {
            int trainingLines = countMap.get(inputFile.getName()) - testSplitSize;
            try {
              assertSplit(inputFile, charset, testSplitSize, trainingLines, tempTrainingDirectory, tempTestDirectory);
            }
            catch (Exception e) {
              throw new RuntimeException(e);
            }
          }
    });
    
    si.splitDirectory(tempInputDirectory);
  }
  
  public void testSplitFile() throws Exception {
    writeSingleInputFile();
    si.setTestSplitSize(2);
    si.setCallback(new TestCallback(2, 10));
    si.splitFile(tempInputFile);
  }
  
  public void testSplitFileLocation() throws Exception {
    writeSingleInputFile();
    si.setTestSplitSize(2);
    si.setSplitLocation(50);
    si.setCallback(new TestCallback(2, 10));
    si.splitFile(tempInputFile);
  }
  
  public void testSplitFilePct() throws Exception {
    writeSingleInputFile();
    si.setTestSplitPct(25);
   
    si.setCallback(new TestCallback(3, 9));
    si.splitFile(tempInputFile);
  }
  
  public void testSplitFilePctLocation() throws Exception {
    writeSingleInputFile();
    si.setTestSplitPct(25);
    si.setSplitLocation(50);
    si.setCallback(new TestCallback(3, 9));
    si.splitFile(tempInputFile);
  }
  
  public void testSplitFileRandomSelectionSize() throws Exception {
    writeSingleInputFile();
    si.setTestRandomSelectionSize(5);
   
    si.setCallback(new TestCallback(5, 7));
    si.splitFile(tempInputFile);
  }
  
  public void testSplitFileRandomSelectionPct() throws Exception {
    writeSingleInputFile();
    si.setTestRandomSelectionPct(25);
   
    si.setCallback(new TestCallback(3, 9));
    si.splitFile(tempInputFile);
  }
  
  public void testValidate() throws Exception {
    SplitBayesInput st = new SplitBayesInput();
    assertValidateException(st, IllegalStateException.class);
    
    st.setTestSplitSize(100);
    assertValidateException(st, NullPointerException.class); 
    
    st.setTestOutputDirectory(tempTestDirectory);
    assertValidateException(st, NullPointerException.class); 
    
    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();
    
    st.setTestSplitPct(50);
    assertValidateException(st, IllegalStateException.class);
    
    st = new SplitBayesInput();
    st.setTestRandomSelectionPct(50);
    st.setTestOutputDirectory(tempTestDirectory);
    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();
    
    st.setTestSplitPct(50);
    assertValidateException(st, IllegalStateException.class);
    
    st = new SplitBayesInput();
    st.setTestRandomSelectionPct(50);
    st.setTestOutputDirectory(tempTestDirectory);
    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();
    
    st.setTestSplitSize(100);
    assertValidateException(st, IllegalStateException.class);
  }
  
  private class TestCallback implements SplitBayesInput.SplitCallback {
    int testSplitSize;
    int trainingLines;
    
    public TestCallback(int testSplitSize, int trainingLines) {
      this.testSplitSize = testSplitSize;
      this.trainingLines = trainingLines;
    }
    
    @Override
    public void splitComplete(File inputFile, int lineCount, int trainCount, int testCount, int testSplitStart) {
      try {
        assertSplit(tempInputFile, charset, testSplitSize, trainingLines, tempTrainingDirectory, tempTestDirectory);
      }
      catch (Exception e) {
        throw new RuntimeException(e);
      }
    }
  }
  
  private void assertValidateException(SplitBayesInput st, Class<?> clazz) {
    try {
      st.validate();
      TestCase.fail("Expected valdate() to throw an exception, received none");
    }
    catch (Exception e) {
      if (!e.getClass().isAssignableFrom(clazz)) {
        e.printStackTrace();
        TestCase.fail("Unexpected exception. Expected " + clazz.getName() + " received " + e.getClass().getName());
      }
    } 
  }
  
  private void assertSplit(File tempInputFile, Charset charset, int testSplitSize, int trainingLines, 
      File tempTrainingDirectory, File tempTestDirectory) throws Exception {
    
    File testFile = new File(tempTestDirectory, tempInputFile.getName());
    TestCase.assertTrue("test file exists", testFile.isFile());
    TestCase.assertEquals("test line count", testSplitSize, SplitBayesInput.countLines(testFile,charset));
    
    File trainingFile = new File(tempTrainingDirectory, tempInputFile.getName());
    TestCase.assertTrue("training file exists", trainingFile.isFile());
    TestCase.assertEquals("training line count", trainingLines, SplitBayesInput.countLines(trainingFile, charset));  
  }
}
