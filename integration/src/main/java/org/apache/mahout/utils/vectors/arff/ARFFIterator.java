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

package org.apache.mahout.utils.vectors.arff;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.List;

import com.google.common.collect.AbstractIterator;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

final class ARFFIterator extends AbstractIterator<Vector> {

  // This pattern will make sure a , inside a string is not a point for split.
  // Ex: "Arizona" , "0:08 PM, PDT" , 110 will be split considering "0:08 PM, PDT" as one string
  private static final Pattern WORDS_WITHOUT_SPARSE = Pattern.compile("([\\w[^{]])*");
  private static final Pattern DATA_PATTERN = Pattern.compile("^\\"+ARFFModel.ARFF_SPARSE+"(.*)\\"+ARFFModel.ARFF_SPARSE_END+"$");

  private final BufferedReader reader;
  private final ARFFModel model;

  ARFFIterator(BufferedReader reader, ARFFModel model) {
    this.reader = reader;
    this.model = model;
  }

  @Override
  protected Vector computeNext() {
    String line;
    try {
      while ((line = reader.readLine()) != null) {
        line = line.trim();
        if (!line.isEmpty() && !line.startsWith(ARFFModel.ARFF_COMMENT)) {
          break;
        }
      }
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
    if (line == null) {
      try {
        Closeables.close(reader, true);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
      return endOfData();
    }
    Vector result;
    Matcher contents = DATA_PATTERN.matcher(line);
    if (contents.find()) {
      line = contents.group(1);
      String[] splits = splitCSV(line);
      result = new RandomAccessSparseVector(model.getLabelSize());
      for (String split : splits) {
        int idIndex = split.indexOf(' ');
        int idx = Integer.parseInt(split.substring(0, idIndex).trim());
        String data = split.substring(idIndex).trim();
        if (!"?".equals(data)) {
          result.setQuick(idx, model.getValue(data, idx));
        }
      }
    } else {
      result = new DenseVector(model.getLabelSize());
      String[] splits = splitCSV(line);
      for (int i = 0; i < splits.length; i++) {
        String split = splits[i];
        split = split.trim();
        if (WORDS_WITHOUT_SPARSE.matcher(split).matches() && !"?".equals(split)) {
          result.setQuick(i, model.getValue(split, i));
        }
      }
    }
    return result;
  }

  /**
   * Splits a string by comma, ignores commas inside quotes and escaped quotes.
   * As quotes are both double and single possible, because there is no exact definition
   * for ARFF files
   * @param line -
   * @return String[]
   */
  public static String[] splitCSV(String line) {
    StringBuilder sb = new StringBuilder(128);
    List<String> tokens = Lists.newArrayList();
    char escapeChar = '\0';
    for (int i = 0; i < line.length(); i++) {
      char c = line.charAt(i);
      if (c == '\\') {
        i++;
        sb.append(line.charAt(i));
      }
      else if (c == '"' || c == '\'') {
        // token is closed
        if (c == escapeChar) {
          escapeChar = '\0';
        }
        else if (escapeChar == '\0') {
          escapeChar = c;
        }
        sb.append(c);
      }
      else if (c == ',') {
        if (escapeChar == '\0') {
          tokens.add(sb.toString().trim());
          sb.setLength(0); // start work on next token
        }
        else {
          sb.append(c);
        }
      }
      else {
        sb.append(c);
      }
    }
    if (sb.length() > 0) {
      tokens.add(sb.toString().trim());
    }

    return tokens.toArray(new String[tokens.size()]);
  }

}
