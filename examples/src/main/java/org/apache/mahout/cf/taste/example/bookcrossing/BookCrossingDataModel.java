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

import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FileLineIterable;
import org.apache.mahout.cf.taste.impl.common.IOUtils;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.example.grouplens.GroupLensDataModel;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.OutputStreamWriter;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.List;
import java.nio.charset.Charset;

public final class BookCrossingDataModel extends FileDataModel {

  private Map<String, Book> bookMap;
  private Map<String, String[]> userDataMap;
  private final File booksFile;
  private final File usersFile;

  public BookCrossingDataModel() throws IOException {
    this(GroupLensDataModel.readResourceToTempFile("/org/apache/mahout/cf/taste/example/bookcrossing/BX-Book-Ratings.csv"),
         GroupLensDataModel.readResourceToTempFile("/org/apache/mahout/cf/taste/example/bookcrossing/BX-Books.csv"),
         GroupLensDataModel.readResourceToTempFile("/org/apache/mahout/cf/taste/example/bookcrossing/BX-Users.csv"));
  }

  /**
   * @param ratingsFile BookCrossing ratings file in its native format
   * @param booksFile BookCrossing books file in its native format
   * @param usersFile BookCrossing books file in its native format
   * @throws IOException if an error occurs while reading or writing files
   */
  public BookCrossingDataModel(File ratingsFile, File booksFile, File usersFile) throws IOException {
    super(convertBCFile(ratingsFile));
    this.booksFile = booksFile;
    this.usersFile = usersFile;
  }

  @Override
  protected void reload() {
    bookMap = new FastMap<String, Book>(5001);
    userDataMap = new FastMap<String, String[]>(5001);

    for (String line : new FileLineIterable(booksFile, true)) {
      String[] tokens = tokenizeLine(line, 5);
      if (tokens != null) {
        String id = tokens[0];
        bookMap.put(id, new Book(id, tokens[1], tokens[2], Integer.parseInt(tokens[3]), tokens[4]));
      }
    }
    for (String line : new FileLineIterable(usersFile, true)) {
      String[] tokens = tokenizeLine(line, 3);
      if (tokens != null) {
        String id = tokens[0];
        userDataMap.put(id, new String[] { tokens[1], tokens[2] });
      }
    }

    super.reload();
    bookMap = null;
    userDataMap = null;
  }

  private static String[] tokenizeLine(String line, int numTokens) {
    String[] result = new String[numTokens];
    int pos = 0;
    int token = 0;
    int start = 0;
    int end = 0;
    boolean inQuote = false;
    int length = line.length();
    while (pos < length && token < numTokens) {
      char c = line.charAt(pos);
      if (c == '"') {
        if (inQuote) {
          if (line.charAt(pos - 1) != '\\') {
            end = pos;
            inQuote = false;
          }
        } else {
          start = pos + 1;
          inQuote = true;
        }
      } else if (c == ';' && !inQuote) {
        if (start == end) {
          // last token was unquoted
          end = pos + 1;
        }
        result[token] = line.substring(start, end);
        start = pos + 1;
        end = pos + 1;
        token++;
      }
      pos++;
    }
    if (token == numTokens - 1) {
      // one more at end
      if (start == end) {
        // last token was unquoted
        end = pos;
      }
      result[token] = line.substring(start, end);
      token++;
    }
    if (token != numTokens) {
      return null;
    }
    for (int i = 0; i < result.length; i++) {
      if ("NULL".equalsIgnoreCase(result[i])) {
        result[i] = null;
      }
    }
    return result;
  }
  
  @Override
  protected User buildUser(String id, List<Preference> prefs) {
    String[] userData = userDataMap.get(id);
    if (userData == null) {
      throw new NoSuchElementException();
    }
    String location = userData[0];
    Integer age = userData[1] == null ? null : Integer.valueOf(userData[1]);
    return new BookCrossingUser(id, prefs, location, age);
  }

  @Override
  protected Item buildItem(String id) {
    Item item = bookMap.get(id);
    if (item == null) {
      // some books aren't in books file?
      return new Book(id, null, null, 0, null);
    }
    return item;
  }

  private static File convertBCFile(File originalFile) throws IOException {
    File resultFile = new File(new File(System.getProperty("java.io.tmpdir")), "taste.bookcrossing.txt");
    if (!resultFile.exists()) {
      PrintWriter writer = null;
      try {
        writer = new PrintWriter(new OutputStreamWriter(new FileOutputStream(resultFile), Charset.forName("UTF-8")));
        for (String line : new FileLineIterable(originalFile, true)) {
          if (line.indexOf(',') >= 0) {
            // crude hack to work around corruptions in data file -- some bad lines with commas in them
            continue;
          }
          String convertedLine = line.replace(';', ',').replace("\"", "");
          writer.println(convertedLine);
        }
        writer.flush();
      } catch (IOException ioe) {
        resultFile.delete();
        throw ioe;
      } finally {
        IOUtils.quietClose(writer);
      }
    }
    return resultFile;
  }

  @Override
  public String toString() {
    return "BookCrossingDataModel";
  }

}