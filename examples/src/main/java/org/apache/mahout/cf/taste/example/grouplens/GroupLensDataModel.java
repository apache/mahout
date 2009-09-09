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

package org.apache.mahout.cf.taste.example.grouplens;

import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.common.IOUtils;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.regex.Pattern;

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
    PrintWriter writer = null;
    try {
      writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(resultFile), Charset.forName("UTF-8")));
      for (String line : new FileLineIterable(originalFile, false)) {
        String convertedLine = COLON_DELIMITER_PATTERN.matcher(line.substring(0, line.lastIndexOf(COLON_DELIMTER))).replaceAll(",");
        writer.println(convertedLine);
      }
      writer.flush();
    } catch (IOException ioe) {
      resultFile.delete();
      throw ioe;
    } finally {
      IOUtils.quietClose(writer);
    }
    return resultFile;
  }

  public static File readResourceToTempFile(String resourceName) throws IOException {
    InputStream is = GroupLensRecommender.class.getResourceAsStream(resourceName);
    if (is == null) {
      // No resource found, try just using the file
      return new File("src/main/java" + resourceName);
    }
    try {
      File tempFile = File.createTempFile("taste", null);
      tempFile.deleteOnExit();
      OutputStream os = new FileOutputStream(tempFile);
      try {
        int bytesRead;
        byte[] buffer = new byte[32768];
        while ((bytesRead = is.read(buffer)) > 0) {
          os.write(buffer, 0, bytesRead);
        }
        os.flush();
        return tempFile;
      } finally {
        IOUtils.quietClose(os);
      }
    } finally {
      IOUtils.quietClose(is);
    }
  }


  @Override
  public String toString() {
    return "GroupLensDataModel";
  }

}
