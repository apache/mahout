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

package org.apache.mahout.vectorizer;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.StringTuple;
import org.junit.Test;

import java.util.Arrays;

/**
 * Tests tokenizing of <Text documentId, Text text> {@link SequenceFile}s by the {@link DocumentProcessor} into
 * <Text documentId, StringTuple tokens> sequence files
 */
public class DocumentProcessorTest extends MahoutTestCase {

  @Test
  public void testTokenizeDocuments() throws Exception {
    Configuration configuration = new Configuration();
    FileSystem fs = FileSystem.get(configuration);
    Path input = new Path(getTestTempDirPath(), "inputDir");
    Path output = new Path(getTestTempDirPath(), "outputDir");

    String documentId1 = "123";
    String documentId2 = "456";

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, configuration, input, Text.class, Text.class);
    try {
      String text1 = "A test for the document processor";
      writer.append(new Text(documentId1), new Text(text1));
      String text2 = "and another one";
      writer.append(new Text(documentId2), new Text(text2));
    } finally {
      Closeables.closeQuietly(writer);
    }

    DocumentProcessor.tokenizeDocuments(input, DefaultAnalyzer.class, output, configuration);

    FileStatus[] statuses = fs.listStatus(output);
    assertEquals(1, statuses.length);
    Path filePath = statuses[0].getPath();
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, filePath, configuration);
    Text key = reader.getKeyClass().asSubclass(Text.class).newInstance();
    StringTuple value = reader.getValueClass().asSubclass(StringTuple.class).newInstance();

    reader.next(key, value);
    assertEquals(documentId1, key.toString());
    assertEquals(Arrays.asList("test", "document", "processor"), value.getEntries());
    reader.next(key, value);
    assertEquals(documentId2, key.toString());
    assertEquals(Arrays.asList("another", "one"), value.getEntries());
  }
}
