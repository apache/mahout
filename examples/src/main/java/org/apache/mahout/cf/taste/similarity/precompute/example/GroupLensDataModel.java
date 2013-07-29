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

package org.apache.mahout.cf.taste.similarity.precompute.example;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.URL;
import java.util.regex.Pattern;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import com.google.common.io.InputSupplier;
import com.google.common.io.Resources;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.common.iterator.FileLineIterable;

public final class GroupLensDataModel extends FileDataModel {
  
  private static final String COLON_DELIMTER = "::";
  private static final Pattern COLON_DELIMITER_PATTERN = Pattern.compile(COLON_DELIMTER);
  
  public GroupLensDataModel() throws IOException {
    this(readResourceToTempFile("/org/apache/mahout/cf/taste/example/grouplens/ratings.dat"));
  }
  
  /**
   * @param ratingsFile GroupLens ratings.dat file in its native format
   * @throws IOException if an error occurs while reading or writing files
   */
  public GroupLensDataModel(File ratingsFile) throws IOException {
    super(convertGLFile(ratingsFile));
  }
  
  private static File convertGLFile(File originalFile) throws IOException {
    // Now translate the file; remove commas, then convert "::" delimiter to comma
    File resultFile = new File(new File(System.getProperty("java.io.tmpdir")), "ratings.txt");
    if (resultFile.exists()) {
      resultFile.delete();
    }
    Writer writer = null;
    try {
      writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8);
      for (String line : new FileLineIterable(originalFile, false)) {
        int lastDelimiterStart = line.lastIndexOf(COLON_DELIMTER);
        if (lastDelimiterStart < 0) {
          throw new IOException("Unexpected input format on line: " + line);
        }
        String subLine = line.substring(0, lastDelimiterStart);
        String convertedLine = COLON_DELIMITER_PATTERN.matcher(subLine).replaceAll(",");
        writer.write(convertedLine);
        writer.write('\n');
      }
    } catch (IOException ioe) {
      resultFile.delete();
      throw ioe;
    } finally {
      Closeables.close(writer, false);
    }
    return resultFile;
  }

  public static File readResourceToTempFile(String resourceName) throws IOException {
    InputSupplier<? extends InputStream> inSupplier;
    try {
      URL resourceURL = Resources.getResource(GroupLensDataModel.class, resourceName);
      inSupplier = Resources.newInputStreamSupplier(resourceURL);
    } catch (IllegalArgumentException iae) {
      File resourceFile = new File("src/main/java" + resourceName);
      inSupplier = Files.newInputStreamSupplier(resourceFile);
    }
    File tempFile = File.createTempFile("taste", null);
    tempFile.deleteOnExit();
    Files.copy(inSupplier, tempFile);
    return tempFile;
  }

  @Override
  public String toString() {
    return "GroupLensDataModel";
  }
  
}
