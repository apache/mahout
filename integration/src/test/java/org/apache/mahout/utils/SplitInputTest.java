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

package org.apache.mahout.utils;

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
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.classifier.ClassifierData;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterable;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;
import org.junit.Before;
import org.junit.Test;

public final class SplitInputTest extends MahoutTestCase {

  private OpenObjectIntHashMap<String> countMap;
  private Charset charset;
  private FileSystem fs;
  private Path tempInputFile;
  private Path tempTrainingDirectory;
  private Path tempTestDirectory;
  private Path tempMapRedOutputDirectory;
  private Path tempInputDirectory;
  private Path tempSequenceDirectory;
  private SplitInput si;

  @Override
  @Before
  public void setUp() throws Exception {
    Configuration conf = getConfiguration();
    fs = FileSystem.get(conf);

    super.setUp();

    countMap = new OpenObjectIntHashMap<String>();

    charset = Charsets.UTF_8;
    tempSequenceDirectory = getTestTempFilePath("tmpsequence");
    tempInputFile = getTestTempFilePath("bayesinputfile");
    tempTrainingDirectory = getTestTempDirPath("bayestrain");
    tempTestDirectory = getTestTempDirPath("bayestest");
    tempMapRedOutputDirectory = new Path(getTestTempDirPath(), "mapRedOutput");
    tempInputDirectory = getTestTempDirPath("bayesinputdir");

    si = new SplitInput();
    si.setTrainingOutputDirectory(tempTrainingDirectory);
    si.setTestOutputDirectory(tempTestDirectory);
    si.setInputDirectory(tempInputDirectory);
  }

  private void writeMultipleInputFiles() throws IOException {
    Writer writer = null;
    String currentLabel = null;
    try {
     for (String[] entry : ClassifierData.DATA) {
      if (!entry[0].equals(currentLabel)) {
        currentLabel = entry[0];
        Closeables.close(writer, false);

        writer = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(tempInputDirectory, currentLabel)),
            Charsets.UTF_8));
      }
      countMap.adjustOrPutValue(currentLabel, 1, 1);
      writer.write(currentLabel + '\t' + entry[1] + '\n');
     }
    }finally {
     Closeables.close(writer, false);
    }
  }

  private void writeSingleInputFile() throws IOException {
    Writer writer = new BufferedWriter(new OutputStreamWriter(fs.create(tempInputFile), Charsets.UTF_8));
    try {
      for (String[] entry : ClassifierData.DATA) {
        writer.write(entry[0] + '\t' + entry[1] + '\n');
      }
    } finally {
      Closeables.close(writer, true);
    }
  }

  @Test
  public void testSplitDirectory() throws Exception {

    writeMultipleInputFiles();

    final int testSplitSize = 1;
    si.setTestSplitSize(testSplitSize);
    si.setCallback(new SplitInput.SplitCallback() {
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

  /**
   * Create a Sequencefile for testing consisting of IntWritable
   * keys and VectorWritable values
   * @param path path for test SequenceFile
   * @param testPoints number of records in test SequenceFile
   */
  private void writeVectorSequenceFile(Path path, int testPoints)
      throws IOException {
    Path tempSequenceFile = new Path(path, "part-00000");
    Configuration conf = getConfiguration();
    IntWritable key = new IntWritable();
    VectorWritable value = new VectorWritable();
    SequenceFile.Writer writer = null;
    try {
      writer =
          SequenceFile.createWriter(fs, conf, tempSequenceFile,
              IntWritable.class, VectorWritable.class);
      for (int i = 0; i < testPoints; i++) {
        key.set(i);
        Vector v = new SequentialAccessSparseVector(4);
        v.assign(i);
        value.set(v);
        writer.append(key, value);
      }
    } finally {
      IOUtils.closeStream(writer);
    }
  }

  /**
   * Create a Sequencefile for testing consisting of IntWritable
   * keys and Text values
   * @param path path for test SequenceFile
   * @param testPoints number of records in test SequenceFile
   */
  private void writeTextSequenceFile(Path path, int testPoints)
      throws IOException {
    Path tempSequenceFile = new Path(path, "part-00000");
    Configuration conf = getConfiguration();
    Text key = new Text();
    Text value = new Text();
    SequenceFile.Writer writer = null;
    try {
      writer =
          SequenceFile.createWriter(fs, conf, tempSequenceFile,
              Text.class, Text.class);
      for (int i = 0; i < testPoints; i++) {
        key.set(Integer.toString(i));
        value.set("Line " + i);
        writer.append(key, value);
      }
    } finally {
      IOUtils.closeStream(writer);
    }
  }

  /**
   * Display contents of a SequenceFile
   * @param sequenceFilePath path to SequenceFile
   */
  private void displaySequenceFile(Path sequenceFilePath) throws IOException {
    for (Pair<?,?> record : new SequenceFileIterable<Writable,Writable>(sequenceFilePath, true, getConfiguration())) {
      System.out.println(record.getFirst() + "\t" + record.getSecond());
    }
  }

  /**
   * Determine number of records in a SequenceFile
   * @param sequenceFilePath path to SequenceFile
   * @return number of records
   */
  private int getNumberRecords(Path sequenceFilePath) throws IOException {
    int numberRecords = 0;
    for (Object value : new SequenceFileValueIterable<Writable>(sequenceFilePath, true, getConfiguration())) {
      numberRecords++;
    }
    return numberRecords;
  }

  /**
   * Test map reduce version of split input with Text, Text key value
   * pairs in input
   */
  @Test
  public void testSplitInputMapReduceText() throws Exception {
    writeTextSequenceFile(tempSequenceDirectory, 1000);
    testSplitInputMapReduce(1000);
  }

  /**
   * Test map reduce version of split input with Text, Text key value
   * pairs in input called from command line
   */
  @Test
  public void testSplitInputMapReduceTextCli() throws Exception {
    writeTextSequenceFile(tempSequenceDirectory, 1000);
    testSplitInputMapReduceCli(1000);
  }

  /**
   * Test map reduce version of split input with IntWritable, Vector key value
   * pairs in input
   */
  @Test
  public void testSplitInputMapReduceVector() throws Exception {
    writeVectorSequenceFile(tempSequenceDirectory, 1000);
    testSplitInputMapReduce(1000);
  }

  /**
   * Test map reduce version of split input with IntWritable, Vector key value
   * pairs in input called from command line
   */
  @Test
  public void testSplitInputMapReduceVectorCli() throws Exception {
    writeVectorSequenceFile(tempSequenceDirectory, 1000);
    testSplitInputMapReduceCli(1000);
  }

  /**
   * Test map reduce version of split input through CLI
   */
  private void testSplitInputMapReduceCli(int numPoints) throws Exception {
    int randomSelectionPct = 25;
    int keepPct = 10;
    String[] args =
        { "--method", "mapreduce", "--input", tempSequenceDirectory.toString(),
            "--mapRedOutputDir", tempMapRedOutputDirectory.toString(),
            "--randomSelectionPct", Integer.toString(randomSelectionPct),
            "--keepPct", Integer.toString(keepPct), "-ow" };
    ToolRunner.run(getConfiguration(), new SplitInput(), args);
    validateSplitInputMapReduce(numPoints, randomSelectionPct, keepPct);
  }

  /**
   * Test map reduce version of split input through method call
   */
  private void testSplitInputMapReduce(int numPoints) throws Exception {
    int randomSelectionPct = 25;
    si.setTestRandomSelectionPct(randomSelectionPct);
    int keepPct = 10;
    si.setKeepPct(keepPct);
    si.setMapRedOutputDirectory(tempMapRedOutputDirectory);
    si.setUseMapRed(true);
    si.splitDirectory(getConfiguration(), tempSequenceDirectory);

    validateSplitInputMapReduce(numPoints, randomSelectionPct, keepPct);
  }

  /**
   * Validate that number of test records and number of training records
   * are consistant with keepPct and randomSelectionPct
   */
  private void validateSplitInputMapReduce(int numPoints, int randomSelectionPct, int keepPct) throws IOException {
    Path testPath = new Path(tempMapRedOutputDirectory, "test-r-00000");
    Path trainingPath = new Path(tempMapRedOutputDirectory, "training-r-00000");
    int numberTestRecords = getNumberRecords(testPath);
    int numberTrainingRecords = getNumberRecords(trainingPath);
    System.out.printf("Test data: %d records\n", numberTestRecords);
    displaySequenceFile(testPath);
    System.out.printf("Training data: %d records\n", numberTrainingRecords);
    displaySequenceFile(trainingPath);
    assertEquals((randomSelectionPct / 100.0) * (keepPct / 100.0) * numPoints,
        numberTestRecords, 2);
    assertEquals(
        (1 - randomSelectionPct / 100.0) * (keepPct / 100.0) * numPoints,
        numberTrainingRecords, 2);
  }

  @Test
  public void testValidate() throws Exception {
    SplitInput st = new SplitInput();
    assertValidateException(st);

    st.setTestSplitSize(100);
    assertValidateException(st);

    st.setTestOutputDirectory(tempTestDirectory);
    assertValidateException(st);

    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();

    st.setTestSplitPct(50);
    assertValidateException(st);

    st = new SplitInput();
    st.setTestRandomSelectionPct(50);
    st.setTestOutputDirectory(tempTestDirectory);
    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();

    st.setTestSplitPct(50);
    assertValidateException(st);

    st = new SplitInput();
    st.setTestRandomSelectionPct(50);
    st.setTestOutputDirectory(tempTestDirectory);
    st.setTrainingOutputDirectory(tempTrainingDirectory);
    st.validate();

    st.setTestSplitSize(100);
    assertValidateException(st);
  }

  private class TestCallback implements SplitInput.SplitCallback {
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

  private static void assertValidateException(SplitInput st) throws IOException {
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
      assertEquals("test line count", testSplitSize, SplitInput.countLines(fs, testFile, charset));

      Path trainingFile = new Path(tempTrainingDirectory, tempInputFile.getName());
      //assertTrue("training file exists", trainingFile.isFile());
      assertEquals("training line count", trainingLines, SplitInput.countLines(fs, trainingFile, charset));
    } catch (IOException ioe) {
      fail(ioe.toString());
    }
  }
}
