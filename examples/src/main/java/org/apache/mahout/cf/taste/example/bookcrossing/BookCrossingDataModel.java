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

package org.apache.mahout.cf.taste.example.bookcrossing;

import org.apache.mahout.cf.taste.example.grouplens.GroupLensDataModel;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
import org.apache.mahout.cf.taste.impl.common.IOUtils;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.nio.charset.Charset;

/**
 * See <a href="http://www.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip">download</a> for
 * data needed by this class. The BX-Book-Ratings.csv file is needed.
 */
public final class BookCrossingDataModel extends FileDataModel {

  public BookCrossingDataModel() throws IOException {
    this(GroupLensDataModel.readResourceToTempFile(
            "/org/apache/mahout/cf/taste/example/bookcrossing/BX-Book-Ratings.csv"));
  }

  /**
   * @param ratingsFile BookCrossing ratings file in its native format
   * @throws IOException if an error occurs while reading or writing files
   */
  public BookCrossingDataModel(File ratingsFile) throws IOException {
    super(convertBCFile(ratingsFile));
  }

  private static File convertBCFile(File originalFile) throws IOException {
    if (!originalFile.exists()) {
      throw new FileNotFoundException(originalFile.toString());
    }
    File resultFile = new File(new File(System.getProperty("java.io.tmpdir")), "taste.bookcrossing.txt");
    resultFile.delete();
    PrintWriter writer = null;
    try {
      writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(resultFile), Charset.forName("UTF-8")));
      for (String line : new FileLineIterable(originalFile, true)) {
        // Delete replace anything that isn't numeric, or a semicolon delimiter. Make comma the delimiter.
        String convertedLine = line.replaceAll("[^0-9;]", "").replace(';', ',');
        // If this means we deleted an entire ID -- few cases like that -- skip the line
        if (convertedLine.contains(",,")) {
          continue;
        }
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

  @Override
  public String toString() {
    return "BookCrossingDataModel";
  }

}