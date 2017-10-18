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

package org.apache.mahout.cf.taste.example.bookcrossing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.regex.Pattern;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.similarity.precompute.example.GroupLensDataModel;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.common.iterator.FileLineIterable;

/**
 * See <a href="http://www.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip">download</a> for
 * data needed by this class. The BX-Book-Ratings.csv file is needed.
 */
public final class BookCrossingDataModel extends FileDataModel {

  private static final Pattern NON_DIGIT_SEMICOLON_PATTERN = Pattern.compile("[^0-9;]");

  public BookCrossingDataModel(boolean ignoreRatings) throws IOException {
    this(GroupLensDataModel.readResourceToTempFile(
             "/org/apache/mahout/cf/taste/example/bookcrossing/BX-Book-Ratings.csv"),
         ignoreRatings);
  }
  
  /**
   * @param ratingsFile BookCrossing ratings file in its native format
   * @throws IOException if an error occurs while reading or writing files
   */
  public BookCrossingDataModel(File ratingsFile, boolean ignoreRatings) throws IOException {
    super(convertBCFile(ratingsFile, ignoreRatings));
  }
  
  private static File convertBCFile(File originalFile, boolean ignoreRatings) throws IOException {
    if (!originalFile.exists()) {
      throw new FileNotFoundException(originalFile.toString());
    }
    File resultFile = new File(new File(System.getProperty("java.io.tmpdir")), "taste.bookcrossing.txt");
    resultFile.delete();
    Writer writer = null;
    try {
      writer = new OutputStreamWriter(new FileOutputStream(resultFile), Charsets.UTF_8);
      for (String line : new FileLineIterable(originalFile, true)) {
        // 0 ratings are basically "no rating", ignore them (thanks h.9000)
        if (line.endsWith("\"0\"")) {
          continue;
        }
        // Delete replace anything that isn't numeric, or a semicolon delimiter. Make comma the delimiter.
        String convertedLine = NON_DIGIT_SEMICOLON_PATTERN.matcher(line)
            .replaceAll("").replace(';', ',');
        // If this means we deleted an entire ID -- few cases like that -- skip the line
        if (convertedLine.contains(",,")) {
          continue;
        }
        if (ignoreRatings) {
          // drop rating
          convertedLine = convertedLine.substring(0, convertedLine.lastIndexOf(','));
        }
        writer.write(convertedLine);
        writer.write('\n');
      }
      writer.flush();
    } catch (IOException ioe) {
      resultFile.delete();
      throw ioe;
    } finally {
      Closeables.close(writer, false);
    }
    return resultFile;
  }
  
  @Override
  public String toString() {
    return "BookCrossingDataModel";
  }
  
}
