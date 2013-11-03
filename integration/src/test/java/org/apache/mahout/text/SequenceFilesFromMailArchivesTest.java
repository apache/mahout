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
import java.io.FileOutputStream;
import java.util.zip.GZIPOutputStream;

import com.google.common.io.Closeables;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * Test case for the SequenceFilesFromMailArchives command-line application.
 */
public final class SequenceFilesFromMailArchivesTest extends MahoutTestCase {

  private File inputDir;

  /**
   * Create the input and output directories needed for testing
   * the SequenceFilesFromMailArchives application.
   */
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    inputDir = getTestTempDir("mail-archives-in");

    // write test mail messages to a gzipped file in a nested directory
    File subDir = new File(inputDir, "subdir");
    subDir.mkdir();
    File gzFile = new File(subDir, "mail-messages.gz");
    GZIPOutputStream gzOut = null;
    try {
      gzOut = new GZIPOutputStream(new FileOutputStream(gzFile));
      gzOut.write(testMailMessages.getBytes("UTF-8"));
      gzOut.finish();
    } finally {
      Closeables.close(gzOut, false);
    }
    
    File subDir2 = new File(subDir, "subsubdir");
    subDir2.mkdir();
    File gzFile2 = new File(subDir2, "mail-messages-2.gz");
    try {
      gzOut = new GZIPOutputStream(new FileOutputStream(gzFile2));
      gzOut.write(testMailMessages.getBytes("UTF-8"));
      gzOut.finish();
    } finally {
      Closeables.close(gzOut, false);
    }    
  }

  @Test
  public void testSequential() throws Exception {

    File outputDir = this.getTestTempDir("mail-archives-out");

    String[] args = {
      "--input", inputDir.getAbsolutePath(),
      "--output", outputDir.getAbsolutePath(),
      "--charset", "UTF-8",
      "--keyPrefix", "TEST",
      "--method", "sequential",
      "--body", "--subject", "--separator", ""
    };

    // run the application's main method
    SequenceFilesFromMailArchives.main(args);

    // app should create a single SequenceFile named "chunk-0" in the output dir
    File expectedChunkFile = new File(outputDir, "chunk-0");
    String expectedChunkPath = expectedChunkFile.getAbsolutePath();
    Assert.assertTrue("Expected chunk file " + expectedChunkPath + " not found!", expectedChunkFile.isFile());

    Configuration conf = getConfiguration();
    SequenceFileIterator<Text, Text> iterator = new SequenceFileIterator<Text, Text>(new Path(expectedChunkPath), true, conf);
    Assert.assertTrue("First key/value pair not found!", iterator.hasNext());
    Pair<Text, Text> record = iterator.next();

    File parentFile = new File(new File(new File("TEST"), "subdir"), "mail-messages.gz");
    Assert.assertEquals(new File(parentFile, testVars[0][0]).toString(), record.getFirst().toString());
    Assert.assertEquals(testVars[0][1] + testVars[0][2], record.getSecond().toString());

    Assert.assertTrue("Second key/value pair not found!", iterator.hasNext());

    record = iterator.next();
    Assert.assertEquals(new File(parentFile, testVars[1][0]).toString(), record.getFirst().toString());
    Assert.assertEquals(testVars[1][1] + testVars[1][2], record.getSecond().toString());

    record = iterator.next();
    File parentFileSubSubDir = new File(new File(new File(new File("TEST"), "subdir"), "subsubdir"), "mail-messages-2.gz");
    Assert.assertEquals(new File(parentFileSubSubDir, testVars[0][0]).toString(), record.getFirst().toString());
    Assert.assertEquals(testVars[0][1] + testVars[0][2], record.getSecond().toString());

    Assert.assertTrue("Second key/value pair not found!", iterator.hasNext());
    record = iterator.next();
    Assert.assertEquals(new File(parentFileSubSubDir, testVars[1][0]).toString(), record.getFirst().toString());
    Assert.assertEquals(testVars[1][1] + testVars[1][2], record.getSecond().toString());

    Assert.assertFalse("Only two key/value pairs expected!", iterator.hasNext());
  }

  @Test
  public void testMapReduce() throws Exception {

    Path tmpDir = getTestTempDirPath();
    Path mrOutputDir = new Path(tmpDir, "mail-archives-out-mr");
    Configuration configuration = getConfiguration();
    FileSystem fs = FileSystem.get(configuration);

    File expectedInputFile = new File(inputDir.toString());

    String[] args = {
      "-Dhadoop.tmp.dir=" + configuration.get("hadoop.tmp.dir"),
      "--input", expectedInputFile.getAbsolutePath(),
      "--output", mrOutputDir.toString(),
      "--charset", "UTF-8",
      "--keyPrefix", "TEST",
      "--method", "mapreduce",
      "--body", "--subject", "--separator", ""
    };

    // run the application's main method
    SequenceFilesFromMailArchives.main(args);

    // app should create a single SequenceFile named "chunk-0" in the output dir
    FileStatus[] fileStatuses = fs.listStatus(mrOutputDir.suffix("/part-m-00000"));
    assertEquals(1, fileStatuses.length); // only one
    assertEquals("part-m-00000", fileStatuses[0].getPath().getName());
    SequenceFileIterator<Text, Text> iterator =
      new SequenceFileIterator<Text, Text>(mrOutputDir.suffix("/part-m-00000"), true, configuration);

    Assert.assertTrue("First key/value pair not found!", iterator.hasNext());
    Pair<Text, Text> record = iterator.next();

    File parentFileSubSubDir = new File(new File(new File(new File("TEST"), "subdir"), "subsubdir"), "mail-messages-2.gz");

    Assert.assertEquals(new File(parentFileSubSubDir, testVars[0][0]).toString(), record.getFirst().toString());
    Assert.assertEquals(testVars[0][1] + testVars[0][2], record.getSecond().toString());
    Assert.assertTrue("Second key/value pair not found!", iterator.hasNext());

    record = iterator.next();
    Assert.assertEquals(new File(parentFileSubSubDir, testVars[1][0]).toString(), record.getFirst().toString());
    Assert.assertEquals(testVars[1][1] + testVars[1][2], record.getSecond().toString());

    // test other file
    File parentFile = new File(new File(new File("TEST"), "subdir"), "mail-messages.gz");
    record = iterator.next();
    Assert.assertEquals(new File(parentFile, testVars[0][0]).toString(), record.getFirst().toString());
    Assert.assertEquals(testVars[0][1] + testVars[0][2], record.getSecond().toString());
    Assert.assertTrue("Second key/value pair not found!", iterator.hasNext());

    record = iterator.next();
    Assert.assertEquals(new File(parentFile, testVars[1][0]).toString(), record.getFirst().toString());
    Assert.assertEquals(testVars[1][1] + testVars[1][2], record.getSecond().toString());
    Assert.assertFalse("Only four key/value pairs expected!", iterator.hasNext());
  }

  // Messages extracted and made anonymous from the ASF mail archives
  private static final String[][] testVars = {
    new String[] {
      "user@example.com",
      "Ant task for JDK1.1 collections build option",
      "\nThis is just a test message\n--\nTesty McTester\n"
    },
    new String[] {
      "somebody@example.com",
      "Problem with build files in several directories",
      "\nHi all,\nThis is another test message.\nRegards,\nAnother Test\n"
    }
  };

  private static final String testMailMessages =
    "From user@example.com  Mon Jul 24 19:13:53 2000\n"
      + "Return-Path: <user@example.com>\n"
      + "Mailing-List: contact ant-user-help@jakarta.apache.org; run by ezmlm\n"
      + "Delivered-To: mailing list ant-user@jakarta.apache.org\n"
      + "Received: (qmail 49267 invoked from network); 24 Jul 2000 19:13:53 -0000\n"
      + "Message-ID: <" + testVars[0][0] + ">\n"
      + "From: \"Testy McTester\" <user@example.com>\n"
      + "To: <ant-user@jakarta.apache.org>\n"
      + "Subject: " + testVars[0][1] + '\n'
      + "Date: Mon, 24 Jul 2000 12:24:56 -0700\n"
      + "MIME-Version: 1.0\n"
      + "Content-Type: text/plain;\n"
      + "  charset=\"Windows-1252\"\n"
      + "Content-Transfer-Encoding: 7bit\n"
      + "X-Spam-Rating: locus.apache.org 1.6.2 0/1000/N\n"
      + testVars[0][2] + '\n'
      + "From somebody@example.com  Wed Jul 26 11:32:16 2000\n"
      + "Return-Path: <somebody@example.com>\n"
      + "Mailing-List: contact ant-user-help@jakarta.apache.org; run by ezmlm\n"
      + "Delivered-To: mailing list ant-user@jakarta.apache.org\n"
      + "Received: (qmail 73966 invoked from network); 26 Jul 2000 11:32:16 -0000\n"
      + "User-Agent: Microsoft-Outlook-Express-Macintosh-Edition/5.02.2022\n"
      + "Date: Wed, 26 Jul 2000 13:32:08 +0200\n"
      + "Subject: " + testVars[1][1] + '\n'
      + "From: Another Test <somebody@example.com>\n"
      + "To: <ant-user@jakarta.apache.org>\n"
      + "Message-Id: <" + testVars[1][0] + ">\n"
      + "Mime-Version: 1.0\n"
      + "Content-Type: text/plain; charset=\"US-ASCII\"\n"
      + "Content-Transfer-Encoding: 7bit\n"
      + "X-Spam-Rating: locus.apache.org 1.6.2 0/1000/N\n"
      + testVars[1][2];
}
