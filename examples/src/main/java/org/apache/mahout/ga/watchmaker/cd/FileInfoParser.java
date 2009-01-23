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

package org.apache.mahout.ga.watchmaker.cd;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.ga.watchmaker.cd.DataSet.Attribute;
import org.apache.mahout.ga.watchmaker.cd.DataSet.NominalAttr;
import org.apache.mahout.ga.watchmaker.cd.DataSet.NumericalAttr;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * Initializes a DataSet using a special format file.<br>
 * The file contains for each attribute one of the following:<br>
 * <ul>
 * <li><code>IGNORED</code><br>
 * if the attribute is ignored</li>
 * <li><code>LABEL, val1, val2, ...</code><br>
 * if the attribute is the label, and its possible values</li>
 * <li><code>CATEGORICAL, val1, val2, ...</code><br>
 * if the attribute is nominal, and its possible values</li>
 * <li><code>NUMERICAL, min, max</code><br>
 * if the attribute is numerical, and its min and max values</li>
 * </ul>
 */
public class FileInfoParser {

  public static final String IGNORED_TOKEN = "IGNORED";

  public  static final String LABEL_TOKEN = "LABEL";

  public  static final String NOMINAL_TOKEN = "CATEGORICAL";

  public static final String NUMERICAL_TOKEN = "NUMERICAL";

  private FileInfoParser() {
  }

  /**
   * Initializes a dataset using an info file.
   * 
   * @param fs file system
   * @param inpath info file
   * @return Initialized Dataset
   */
  public static DataSet parseFile(FileSystem fs, Path inpath)
      throws IOException {
    Path info = getInfoFile(fs, inpath);

    FSDataInputStream input = fs.open(info);
    Scanner reader = new Scanner(input);

    List<Integer> ignored = new ArrayList<Integer>();
    List<Attribute> attributes = new ArrayList<Attribute>();
    int labelIndex = -1;

    int index = 0;

    while (reader.hasNextLine()) {
      String line = reader.nextLine();
      StringTokenizer tokenizer = new StringTokenizer(line, ", ");
      String token = nextToken(tokenizer);
      if (IGNORED_TOKEN.equals(token)) {
        ignored.add(index);
      } else if (LABEL_TOKEN.equals(token)) {
        labelIndex = index;
        attributes.add(parseNominal(tokenizer));
      } else if (NOMINAL_TOKEN.equals(token)) {
        attributes.add(parseNominal(tokenizer));
      } else if (NUMERICAL_TOKEN.equals(token)) {
        attributes.add(parseNumerical(tokenizer));
      } else {
        throw new RuntimeException("Unknown token (" + token
            + ") encountered while parsing the info file");
      }
    }

    reader.close();

    if (labelIndex == -1)
      throw new RuntimeException("Info file does not contain a LABEL");

    return new DataSet(attributes, ignored, labelIndex);

  }

  /**
   * Prepares the path for the info file corresponding to the input path.
   * 
   * @param fs file system
   * @param inpath
   * @throws IOException
   */
  public static Path getInfoFile(FileSystem fs, Path inpath)
      throws IOException {
    assert inpath != null : "null inpath parameter";
    if (!fs.exists(inpath))
      throw new RuntimeException("Input path does not exist");
    if (!fs.getFileStatus(inpath).isDir())
      throw new RuntimeException("Input path should be a directory");

    // info file name
    Path infoPath = new Path(inpath.getParent(), inpath.getName() + ".infos");
    if (!fs.exists(infoPath))
      throw new RuntimeException("Info file does not exist");

    return infoPath;
  }

  /**
   * Parse a nominal attribute.
   * 
   * @param tokenizer
   */
  private static NominalAttr parseNominal(StringTokenizer tokenizer) {
    List<String> vlist = new ArrayList<String>();
    while (tokenizer.hasMoreTokens())
      vlist.add(nextToken(tokenizer));

    String[] values = new String[vlist.size()];
    vlist.toArray(values);

    return new NominalAttr(values);
  }

  /**
   * Parse a numerical attribute.
   * 
   * @param tokenizer
   */
  private static NumericalAttr parseNumerical(StringTokenizer tokenizer) {
    double min = nextDouble(tokenizer);
    double max = nextDouble(tokenizer);
    assert min <= max : "min > max";

    return new NumericalAttr(min, max);
  }

  private static double nextDouble(StringTokenizer tokenizer) {
    String token = nextToken(tokenizer);
    double value;

    try {
      value = Double.parseDouble(token);
    } catch (NumberFormatException e) {
      throw new RuntimeException("Exception while parsing info file", e);
    }

    return value;
  }

  private static String nextToken(StringTokenizer tokenizer) {
    String token;
    try {
      token = tokenizer.nextToken();
    } catch (NoSuchElementException e) {
      throw new RuntimeException("Exception while parsing info file", e);
    }

    return token;
  }

}
