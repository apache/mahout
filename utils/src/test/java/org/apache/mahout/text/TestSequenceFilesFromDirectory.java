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

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.utils.MahoutTestCase;
import org.junit.Before;
import org.junit.Test;

public final class TestSequenceFilesFromDirectory extends MahoutTestCase {

  private static final Charset UTF8 = Charset.forName("UTF-8");
  private static final String[][] DATA1 = {
      {"test1", "This is the first text."},
      {"test2", "This is the second text."},
      {"test3", "This is the third text."}
  };

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
  }
  
  /** Story converting text files to SequenceFile */
  @Test
  public void testSequnceFileFromTsvBasic() throws Exception {
    // parameters
    Configuration conf = new Configuration();
    
    FileSystem fs = FileSystem.get(conf);
    
    // create
    Path tmpDir = this.getTestTempDirPath();
    Path inputDir = new Path(tmpDir, "inputDir");
    fs.mkdirs(inputDir);
    Path outputDir = new Path(tmpDir, "outputDir");
    
    // prepare input files
    createFilesFromArrays(conf, inputDir, DATA1);

    String prefix = "UID";
    SequenceFilesFromDirectory.main(new String[] {"--input",
        inputDir.toString(), "--output", outputDir.toString(), "--chunkSize",
        "64", "--charset",
        UTF8.displayName(Locale.ENGLISH), "--keyPrefix", prefix});
    
    // check output chunk files
    checkChunkFiles(conf, outputDir, DATA1, prefix);
  }

  private static void createFilesFromArrays(Configuration conf, Path inputDir, String[][] data) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    for (String[] aData : data) {
      OutputStreamWriter osw = new OutputStreamWriter(fs.create(new Path(inputDir, aData[0])), UTF8);
      osw.write(aData[1]);
      osw.close();
    }
  }

  private static void checkChunkFiles(Configuration conf, Path outputDir, String[][] data, String prefix)
    throws IOException, InstantiationException, IllegalAccessException {
    FileSystem fs = FileSystem.get(conf);
    
    // output exists?
    FileStatus[] fstats = fs.listStatus(outputDir, new ExcludeDotFiles());
    assertEquals(1, fstats.length); // only one
    assertEquals("chunk-0", fstats[0].getPath().getName());
    
    // read a chunk to check content
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, fstats[0].getPath(), conf);
    assertEquals("org.apache.hadoop.io.Text", reader.getKeyClassName());
    assertEquals("org.apache.hadoop.io.Text", reader.getValueClassName());
    Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
    Writable value = reader.getValueClass().asSubclass(Writable.class).newInstance();
    
    Map<String,String> fileToData = new HashMap<String,String>();
    for (String[] aData : data) {
      fileToData.put(prefix + Path.SEPARATOR + aData[0], aData[1]);
    }

    for (String[] aData : data) {
      assertTrue(reader.next(key, value));
      String retrievedData = fileToData.get(key.toString().trim());
      assertNotNull(retrievedData);
      assertEquals(retrievedData, value.toString().trim());
    }
    reader.close();
  }
  
  /**
   * exclude hidden (starting with dot) files
   */
  private static class ExcludeDotFiles implements PathFilter {
    @Override
    public boolean accept(Path file) {
      return !file.getName().startsWith(".");
    }
  }

}

