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

package org.apache.mahout.ga.watchmaker.cd;

import java.io.IOException;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.Lists;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/**
 * Initializes a DataSet using a special format file.<br>
 * The file contains for each attribute one of the following:<br>
 * <ul>
 * <li>{@code IGNORED}<br>
 * if the attribute is ignored</li>
 * <li>{@code LABEL, val1, val2, ...}<br>
 * if the attribute is the label, and its possible values</li>
 * <li>{@code CATEGORICAL, val1, val2, ...}<br>
 * if the attribute is nominal, and its possible values</li>
 * <li>{@code NUMERICAL, min, max}<br>
 * if the attribute is numerical, and its min and max values</li>
 * </ul>
 */
public final class FileInfoParser {
  
  public static final String IGNORED_TOKEN = "IGNORED";
  public static final String LABEL_TOKEN = "LABEL";
  public static final String NOMINAL_TOKEN = "CATEGORICAL";
  public static final String NUMERICAL_TOKEN = "NUMERICAL";
  private static final Splitter COMMA = Splitter.on(',').trimResults();

  private FileInfoParser() { }
  
  /**
   * Initializes a dataset using an info file.
   * 
   * @param fs
   *          file system
   * @param inpath
   *          info file
   * @return Initialized Dataset
   */
  public static DataSet parseFile(FileSystem fs, Path inpath) throws IOException {
    Path info = getInfoFile(fs, inpath);
    
    FSDataInputStream input = fs.open(info);
    Scanner reader = new Scanner(input);
    
    List<Integer> ignored = Lists.newArrayList();
    List<Attribute> attributes = Lists.newArrayList();
    int labelIndex = -1;
    
    int index = 0;
    
    while (reader.hasNextLine()) {
      String line = reader.nextLine();
      Iterator<String> tokens = COMMA.split(line).iterator();
      String token = tokens.next();
      if (IGNORED_TOKEN.equals(token)) {
        ignored.add(index);
      } else if (LABEL_TOKEN.equals(token)) {
        labelIndex = index;
        attributes.add(parseNominal(tokens));
      } else if (NOMINAL_TOKEN.equals(token)) {
        attributes.add(parseNominal(tokens));
      } else if (NUMERICAL_TOKEN.equals(token)) {
        attributes.add(parseNumerical(tokens));
      } else {
        throw new IllegalArgumentException("Unknown token (" + token
                                           + ") encountered while parsing the info file");
      }
    }
    
    reader.close();
    
    if (labelIndex == -1) {
      throw new IllegalStateException("Info file does not contain a LABEL");
    }
    
    return new DataSet(attributes, ignored, labelIndex);
    
  }
  
  /**
   * Prepares the path for the info file corresponding to the input path.
   * 
   * @param fs file system
   */
  public static Path getInfoFile(FileSystem fs, Path inpath) throws IOException {
    Preconditions.checkArgument(inpath != null && fs.exists(inpath) && fs.getFileStatus(inpath).isDir(),
        "Input path should be a directory", inpath);
    Path infoPath = new Path(inpath.getParent(), inpath.getName() + ".infos");
    Preconditions.checkArgument(fs.exists(infoPath), "Info file does not exist", infoPath);
    return infoPath;
  }
  
  /**
   * Parse a nominal attribute.
   */
  private static NominalAttr parseNominal(Iterator<String> tokens) {
    Collection<String> vlist = Lists.newArrayList();
    while (tokens.hasNext()) {
      vlist.add(tokens.next());
    }
    
    String[] values = new String[vlist.size()];
    vlist.toArray(values);
    
    return new NominalAttr(values);
  }
  
  /**
   * Parse a numerical attribute.
   */
  private static NumericalAttr parseNumerical(Iterator<String> tokens) {
    double min = Double.parseDouble(tokens.next());
    double max = Double.parseDouble(tokens.next());
    Preconditions.checkArgument(min <= max, "min > max");
    return new NumericalAttr(min, max);
  }
  
}
