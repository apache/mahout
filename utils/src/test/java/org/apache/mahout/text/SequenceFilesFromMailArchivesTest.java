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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.zip.GZIPOutputStream;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.Text;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Test case for the SequenceFilesFromMailArchives command-line application.
 */
public class SequenceFilesFromMailArchivesTest {
  
  // TODO: Negative tests

  private File inputDir = null;
  private File outputDir = null;

  /**
   * Create the input and output directories needed for testing
   * the SequenceFilesFromMailArchives application.
   */
  @Before
  public void setupBeforeTesting() throws IOException {
    // tread-lightly, create folder names using the timestamp
    long now = System.currentTimeMillis();
    inputDir = createTempDir("mail-archives-"+now+"-in");
    outputDir = createTempDir("mail-archives-"+now+"-out");
    
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
      if (gzOut != null) {
        try {
          gzOut.close();
        } catch (Exception ignore) {}
      }
    }    
  }

  /**
   * Test the main method of the SequenceFilesFromMailArchives
   * command-line application.
   */
  @Test
  public void testMain() throws Exception {
    String[] args = new String[] {
      "--input", inputDir.getAbsolutePath(),  
      "--output", outputDir.getAbsolutePath(),
      "--charset", "UTF-8",
      "--keyPrefix", "TEST"
    };
    
    // run the application's main method
    SequenceFilesFromMailArchives.main(args);
    
    // app should create a single SequenceFile named "chunk-0"
    // in the output dir
    File expectedChunkFile = new File(outputDir, "chunk-0");
    String expectedChunkPath = expectedChunkFile.getAbsolutePath();
    assertTrue("Expected chunk file "+expectedChunkPath+" not found!", 
        expectedChunkFile.isFile());

    Text key = new Text();
    Text value = new Text();
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader seqFileReader = null;
    try {
      seqFileReader = new SequenceFile.Reader(fs, new Path(expectedChunkPath), conf);
      assertTrue("First key/value pair not found!", seqFileReader.next(key, value));
      
      assertEquals("TEST/subdir/mail-messages.gz/"+testVars[0][0], key.toString());
      assertEquals(testVars[0][1]+testVars[0][2], value.toString());
  
      assertTrue("Second key/value pair not found!", seqFileReader.next(key, value));
      assertEquals("TEST/subdir/mail-messages.gz/"+testVars[1][0], key.toString());
      assertEquals(testVars[1][1]+testVars[1][2], value.toString());
  
      assertFalse("Only two key/value pairs expected!", seqFileReader.next(key, value));
    } finally {
      if (seqFileReader != null) {
        try {
          seqFileReader.close();
        } catch (Exception ignore) {}
      }
    }
  }

  @After
  public void cleanupAfterTesting() {
    if (inputDir != null)
      rmdir(inputDir);
    
    if (outputDir != null)
      rmdir(outputDir);
  }

  // creates a temp directory for storing test input / output
  // fails if the directory cannot be created
  private File createTempDir(String dirName) {
    File tempDir = new File(System.getProperty("java.io.tmpdir"), dirName);
    if (!tempDir.isDirectory()) {
      tempDir.mkdirs();
      if (!tempDir.isDirectory()) {
        fail("Failed to create temp directory "+tempDir.getAbsolutePath());
      }
    }
    return tempDir;
  }

  // recursively delete the temp directories created by this test
  private void rmdir(File dir) {
    if (dir.isDirectory()) {
      File[] files = dir.listFiles();
      for (int f=0; f<files.length; f++) {
        if (files[f].isDirectory()) {
          rmdir(files[f]);
        } else {
          files[f].delete();
        }
      }
    }
    dir.delete();
  }
  
  // Messages extracted and anonymized from the ASF mail archives
  private static final String[][] testVars = new String[][] {
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
    "From user@example.com  Mon Jul 24 19:13:53 2000\n"+
    "Return-Path: <user@example.com>\n"+
    "Mailing-List: contact ant-user-help@jakarta.apache.org; run by ezmlm\n"+
    "Delivered-To: mailing list ant-user@jakarta.apache.org\n"+
    "Received: (qmail 49267 invoked from network); 24 Jul 2000 19:13:53 -0000\n"+
    "Message-ID: <"+testVars[0][0]+">\n"+
    "From: \"Testy McTester\" <user@example.com>\n"+
    "To: <ant-user@jakarta.apache.org>\n"+
    "Subject: "+testVars[0][1]+"\n"+
    "Date: Mon, 24 Jul 2000 12:24:56 -0700\n"+
    "MIME-Version: 1.0\n"+
    "Content-Type: text/plain;\n"+
    "  charset=\"Windows-1252\"\n"+
    "Content-Transfer-Encoding: 7bit\n"+
    "X-Spam-Rating: locus.apache.org 1.6.2 0/1000/N\n"+
    testVars[0][2]+
    "\n"+
    "From somebody@example.com  Wed Jul 26 11:32:16 2000\n"+
    "Return-Path: <somebody@example.com>\n"+
    "Mailing-List: contact ant-user-help@jakarta.apache.org; run by ezmlm\n"+
    "Delivered-To: mailing list ant-user@jakarta.apache.org\n"+
    "Received: (qmail 73966 invoked from network); 26 Jul 2000 11:32:16 -0000\n"+
    "User-Agent: Microsoft-Outlook-Express-Macintosh-Edition/5.02.2022\n"+
    "Date: Wed, 26 Jul 2000 13:32:08 +0200\n"+
    "Subject: "+testVars[1][1]+"\n"+
    "From: Another Test <somebody@example.com>\n"+
    "To: <ant-user@jakarta.apache.org>\n"+
    "Message-Id: <"+testVars[1][0]+">\n"+
    "Mime-Version: 1.0\n"+
    "Content-Type: text/plain; charset=\"US-ASCII\"\n"+
    "Content-Transfer-Encoding: 7bit\n"+
    "X-Spam-Rating: locus.apache.org 1.6.2 0/1000/N\n"+
    testVars[1][2];
}
