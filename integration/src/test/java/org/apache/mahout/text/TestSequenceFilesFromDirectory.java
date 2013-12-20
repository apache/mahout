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

package org.apache.mahout.text;

import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Map;

import com.google.common.base.Charsets;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TestSequenceFilesFromDirectory extends MahoutTestCase {

  private static final Logger logger = LoggerFactory.getLogger(TestSequenceFilesFromDirectory.class);

  private static final String[][] DATA1 = {
    {"test1", "This is the first text."},
    {"test2", "This is the second text."},
    {"test3", "This is the third text."}
  };

  private static final String[][] DATA2 = {
    {"recursive_test1", "This is the first text."},
    {"recursive_test2", "This is the second text."},
    {"recursive_test3", "This is the third text."}
  };

  @Test
  public void testSequenceFileFromDirectoryBasic() throws Exception {
    // parameters
    Configuration configuration = getConfiguration();

    FileSystem fs = FileSystem.get(configuration);

    // create
    Path tmpDir = this.getTestTempDirPath();
    Path inputDir = new Path(tmpDir, "inputDir");
    fs.mkdirs(inputDir);

    Path outputDir = new Path(tmpDir, "outputDir");
    Path outputDirRecursive = new Path(tmpDir, "outputDirRecursive");

    Path inputDirRecursive = new Path(tmpDir, "inputDirRecur");
    fs.mkdirs(inputDirRecursive);

    // prepare input files
    createFilesFromArrays(configuration, inputDir, DATA1);

    SequenceFilesFromDirectory.main(new String[]{
      "--input", inputDir.toString(),
      "--output", outputDir.toString(),
      "--chunkSize", "64",
      "--charset", Charsets.UTF_8.name(),
      "--keyPrefix", "UID",
      "--method", "sequential"});

    // check output chunk files
    checkChunkFiles(configuration, outputDir, DATA1, "UID");

    createRecursiveDirFilesFromArrays(configuration, inputDirRecursive, DATA2);

    FileStatus fstInputPath = fs.getFileStatus(inputDirRecursive);
    String dirs = HadoopUtil.buildDirList(fs, fstInputPath);

    System.out.println("\n\n ----- recursive dirs: " + dirs);
    SequenceFilesFromDirectory.main(new String[]{
      "--input", inputDirRecursive.toString(),
      "--output", outputDirRecursive.toString(),
      "--chunkSize", "64",
      "--charset", Charsets.UTF_8.name(),
      "--keyPrefix", "UID",
      "--method", "sequential"});

    checkRecursiveChunkFiles(configuration, outputDirRecursive, DATA2, "UID");
  }

  @Test
  public void testSequenceFileFromDirectoryMapReduce() throws Exception {

    Configuration conf = getConfiguration();

    FileSystem fs = FileSystem.get(conf);

    // create
    Path tmpDir = this.getTestTempDirPath();
    Path inputDir = new Path(tmpDir, "inputDir");
    fs.mkdirs(inputDir);

    Path inputDirRecur = new Path(tmpDir, "inputDirRecur");
    fs.mkdirs(inputDirRecur);

    Path mrOutputDir = new Path(tmpDir, "mrOutputDir");
    Path mrOutputDirRecur = new Path(tmpDir, "mrOutputDirRecur");

    createFilesFromArrays(conf, inputDir, DATA1);

    SequenceFilesFromDirectory.main(new String[]{
      "-Dhadoop.tmp.dir=" + conf.get("hadoop.tmp.dir"),
      "--input", inputDir.toString(),
      "--output", mrOutputDir.toString(),
      "--chunkSize", "64",
      "--charset", Charsets.UTF_8.name(),
      "--method", "mapreduce",
      "--keyPrefix", "UID",
      "--fileFilterClass", "org.apache.mahout.text.TestPathFilter"
    });

    checkMRResultFiles(conf, mrOutputDir, DATA1, "UID");

    createRecursiveDirFilesFromArrays(conf, inputDirRecur, DATA2);

    FileStatus fst_input_path = fs.getFileStatus(inputDirRecur);
    String dirs = HadoopUtil.buildDirList(fs, fst_input_path);

    logger.info("\n\n ---- recursive dirs: {}", dirs);

    SequenceFilesFromDirectory.main(new String[]{
      "-Dhadoop.tmp.dir=" + conf.get("hadoop.tmp.dir"),
      "--input", inputDirRecur.toString(),
      "--output", mrOutputDirRecur.toString(),
      "--chunkSize", "64",
      "--charset", Charsets.UTF_8.name(),
      "--method", "mapreduce",
      "--keyPrefix", "UID",
      "--fileFilterClass", "org.apache.mahout.text.TestPathFilter"
    });

    checkMRResultFilesRecursive(conf, mrOutputDirRecur, DATA2, "UID");
  }


  private static void createFilesFromArrays(Configuration conf, Path inputDir, String[][] data) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    OutputStreamWriter writer;
    for (String[] aData : data) {
      writer = new OutputStreamWriter(fs.create(new Path(inputDir, aData[0])), Charsets.UTF_8);
      try {
        writer.write(aData[1]);
      } finally {
        Closeables.close(writer, false);
      }
    }
  }

  private static void createRecursiveDirFilesFromArrays(Configuration configuration, Path inputDir,
                                                        String[][] data) throws IOException {
    FileSystem fs = FileSystem.get(configuration);

    logger.info("creativeRecursiveDirFilesFromArrays > based on: {}", inputDir.toString());
    Path curPath;
    String currentRecursiveDir = inputDir.toString();

    for (String[] aData : data) {
      OutputStreamWriter writer;

      currentRecursiveDir += "/" + aData[0];
      File subDir = new File(currentRecursiveDir);
      subDir.mkdir();

      curPath = new Path(subDir.toString(), "file.txt");
      writer = new OutputStreamWriter(fs.create(curPath), Charsets.UTF_8);

      logger.info("Created file: {}", curPath.toString());

      try {
        writer.write(aData[1]);
      } finally {
        Closeables.close(writer, false);
      }
    }
  }

  private static void checkChunkFiles(Configuration configuration,
                                      Path outputDir,
                                      String[][] data,
                                      String prefix) throws IOException {
    FileSystem fs = FileSystem.get(configuration);

    // output exists?
    FileStatus[] fileStatuses = fs.listStatus(outputDir, PathFilters.logsCRCFilter());
    assertEquals(1, fileStatuses.length); // only one
    assertEquals("chunk-0", fileStatuses[0].getPath().getName());

    Map<String, String> fileToData = Maps.newHashMap();
    for (String[] aData : data) {
      fileToData.put(prefix + Path.SEPARATOR + aData[0], aData[1]);
    }

    // read a chunk to check content
    SequenceFileIterator<Text, Text> iterator =
      new SequenceFileIterator<Text, Text>(fileStatuses[0].getPath(), true, configuration);
    try {
      while (iterator.hasNext()) {
        Pair<Text, Text> record = iterator.next();
        String retrievedData = fileToData.get(record.getFirst().toString().trim());
        assertNotNull(retrievedData);
        assertEquals(retrievedData, record.getSecond().toString().trim());
      }
    } finally {
      Closeables.close(iterator, true);
    }
  }

  private static void checkRecursiveChunkFiles(Configuration configuration,
                                               Path outputDir,
                                               String[][] data,
                                               String prefix) throws IOException {
    FileSystem fs = FileSystem.get(configuration);

    System.out.println(" ----------- check_Recursive_ChunkFiles ------------");

    // output exists?
    FileStatus[] fileStatuses = fs.listStatus(outputDir, PathFilters.logsCRCFilter());
    assertEquals(1, fileStatuses.length); // only one
    assertEquals("chunk-0", fileStatuses[0].getPath().getName());


    Map<String, String> fileToData = Maps.newHashMap();
    String currentPath = prefix;
    for (String[] aData : data) {
      currentPath += Path.SEPARATOR + aData[0];
      fileToData.put(currentPath + Path.SEPARATOR + "file.txt", aData[1]);
    }

    // read a chunk to check content
    SequenceFileIterator<Text, Text> iterator = new SequenceFileIterator<Text, Text>(fileStatuses[0].getPath(), true, configuration);
    try {
      while (iterator.hasNext()) {
        Pair<Text, Text> record = iterator.next();
        String retrievedData = fileToData.get(record.getFirst().toString().trim());
        System.out.printf("%s >> %s\n", record.getFirst().toString().trim(), record.getSecond().toString().trim());

        assertNotNull(retrievedData);
        assertEquals(retrievedData, record.getSecond().toString().trim());
        System.out.printf(">>> k: %s, v: %s\n", record.getFirst().toString(), record.getSecond().toString());
      }
    } finally {
      Closeables.close(iterator, true);
    }
  }

  private static void checkMRResultFiles(Configuration conf, Path outputDir,
                                         String[][] data, String prefix) throws IOException {
    FileSystem fs = FileSystem.get(conf);

    // output exists?
    FileStatus[] fileStatuses = fs.listStatus(outputDir.suffix("/part-m-00000"), PathFilters.logsCRCFilter());
    assertEquals(1, fileStatuses.length); // only one
    assertEquals("part-m-00000", fileStatuses[0].getPath().getName());
    Map<String, String> fileToData = Maps.newHashMap();
    for (String[] aData : data) {
      System.out.printf("map.put: %s %s\n", prefix + Path.SEPARATOR + aData[0], aData[1]);
      fileToData.put(prefix + Path.SEPARATOR + aData[0], aData[1]);
    }

    // read a chunk to check content
    SequenceFileIterator<Text, Text> iterator = new SequenceFileIterator<Text, Text>(
      fileStatuses[0].getPath(), true, conf);
    try {
      while (iterator.hasNext()) {
        Pair<Text, Text> record = iterator.next();
        String retrievedData = fileToData.get(record.getFirst().toString().trim());

        System.out.printf("MR> %s >> %s\n", record.getFirst().toString().trim(), record.getSecond().toString().trim());
        assertNotNull(retrievedData);
        assertEquals(retrievedData, record.getSecond().toString().trim());
      }
    } finally {
      Closeables.close(iterator, true);
    }
  }

  private static void checkMRResultFilesRecursive(Configuration configuration, Path outputDir,
                                                  String[][] data, String prefix) throws IOException {
    FileSystem fs = FileSystem.get(configuration);

    // output exists?
    FileStatus[] fileStatuses = fs.listStatus(outputDir.suffix("/part-m-00000"), PathFilters.logsCRCFilter());
    assertEquals(1, fileStatuses.length); // only one
    assertEquals("part-m-00000", fileStatuses[0].getPath().getName());
    Map<String, String> fileToData = Maps.newHashMap();
    String currentPath = prefix;

    for (String[] aData : data) {
      currentPath += Path.SEPARATOR + aData[0];
      fileToData.put(currentPath + Path.SEPARATOR + "file.txt", aData[1]);
    }

    // read a chunk to check content
    SequenceFileIterator<Text, Text> iterator = new SequenceFileIterator<Text, Text>(
      fileStatuses[0].getPath(), true, configuration);
    try {
      while (iterator.hasNext()) {
        Pair<Text, Text> record = iterator.next();
        System.out.printf("MR-Recur > Trying to check: %s\n", record.getFirst().toString().trim());
        String retrievedData = fileToData.get(record.getFirst().toString().trim());
        assertNotNull(retrievedData);
        assertEquals(retrievedData, record.getSecond().toString().trim());
      }
    } finally {
      Closeables.close(iterator, true);
    }
  }
}

