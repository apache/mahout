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

package org.apache.mahout.df.data;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;
import java.util.StringTokenizer;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.df.data.Dataset.Attribute;
import org.apache.mahout.math.DenseVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Converts the input data to a Vector Array using the information given by the Dataset.<br>
 * Generates for each line a Vector that contains :<br>
 * <ul>
 * <li>double parsed value for NUMERICAL attributes</li>
 * <li>int value for CATEGORICAL and LABEL attributes</li>
 * </ul>
 * <br>
 * adds an IGNORED first attribute that will contain a unique id for each instance, which is the line number
 * of the instance in the input data
 */
public final class DataLoader {
  
  private static final Logger log = LoggerFactory.getLogger(DataLoader.class);
  
  private DataLoader() { }
  
  /**
   * Converts a comma-separated String to a Vector.
   * 
   * @param id
   *          unique id for the current instance
   * @param attrs
   *          attributes description
   * @param values
   *          used to convert CATEGORICAL attribute values to Integer
   * @return null if there are missing values '?'
   */
  private static Instance parseString(int id, Attribute[] attrs, List<String>[] values, String string) {
    StringTokenizer tokenizer = new StringTokenizer(string, ", ");
    Preconditions.checkArgument(tokenizer.countTokens() == attrs.length, "Wrong number of attributes in the string");

    // extract tokens and check is there is any missing value
    String[] tokens = new String[attrs.length];
    for (int attr = 0; attr < attrs.length; attr++) {
      String token = tokenizer.nextToken();
      
      if (attrs[attr].isIgnored()) {
        continue;
      }
      
      if ("?".equals(token)) {
        return null; // missing value
      }
      
      tokens[attr] = token;
    }
    
    int nbattrs = Dataset.countAttributes(attrs);
    
    DenseVector vector = new DenseVector(nbattrs);
    
    int aId = 0;
    int label = -1;
    for (int attr = 0; attr < attrs.length; attr++) {
      if (attrs[attr].isIgnored()) {
        continue;
      }
      
      String token = tokens[attr];
      
      if (attrs[attr].isNumerical()) {
        vector.set(aId++, Double.parseDouble(token));
      } else { // CATEGORICAL or LABEL
        // update values
        if (values[attr] == null) {
          values[attr] = Lists.newArrayList();
        }
        if (!values[attr].contains(token)) {
          values[attr].add(token);
        }
        
        if (attrs[attr].isCategorical()) {
          vector.set(aId++, values[attr].indexOf(token));
        } else { // LABEL
          label = values[attr].indexOf(token);
        }
      }
    }
    
    if (label == -1) {
      throw new IllegalStateException("Label not found!");
    }
    
    return new Instance(id, vector, label);
  }
  
  /**
   * Loads the data from a file
   * 
   * @param fs
   *          file system
   * @param fpath
   *          data file path
   * @throws IOException
   *           if any problem is encountered
   */
  
  public static Data loadData(Dataset dataset, FileSystem fs, Path fpath) throws IOException {
    FSDataInputStream input = fs.open(fpath);
    Scanner scanner = new Scanner(input);
    
    List<Instance> instances = Lists.newArrayList();
    
    DataConverter converter = new DataConverter(dataset);
    
    while (scanner.hasNextLine()) {
      String line = scanner.nextLine();
      if (line.isEmpty()) {
        log.warn("{}: empty string", instances.size());
        continue;
      }
      
      Instance instance = converter.convert(instances.size(), line);
      if (instance == null) {
        // missing values found
        log.warn("{}: missing values", instances.size());
        continue;
      }
      
      instances.add(instance);
    }
    
    scanner.close();
    
    return new Data(dataset, instances);
  }
  
  /**
   * Loads the data from a String array
   */
  public static Data loadData(Dataset dataset, String[] data) {
    List<Instance> instances = Lists.newArrayList();
    
    DataConverter converter = new DataConverter(dataset);
    
    for (String line : data) {
      if (line.isEmpty()) {
        log.warn("{}: empty string", instances.size());
        continue;
      }
      
      Instance instance = converter.convert(instances.size(), line);
      if (instance == null) {
        // missing values found
        log.warn("{}: missing values", instances.size());
        continue;
      }
      
      instances.add(instance);
    }
    
    return new Data(dataset, instances);
  }
  
  /**
   * Generates the Dataset by parsing the entire data
   * 
   * @param descriptor
   *          attributes description
   * @param fs
   *          file system
   * @param path
   *          data path
   */
  public static Dataset generateDataset(String descriptor, FileSystem fs, Path path) throws DescriptorException,
                                                                                    IOException {
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);
    
    FSDataInputStream input = fs.open(path);
    Scanner scanner = new Scanner(input);
    
    // used to convert CATEGORICAL attribute to Integer
    List<String>[] values = new List[attrs.length];
    
    int id = 0;
    while (scanner.hasNextLine()) {
      String line = scanner.nextLine();
      if (line.isEmpty()) {
        continue;
      }
      
      if (parseString(id, attrs, values, line) != null) {
        id++;
      }
    }
    
    scanner.close();
    
    return new Dataset(attrs, values, id);
  }
  
  /**
   * Generates the Dataset by parsing the entire data
   * 
   * @param descriptor
   *          attributes description
   */
  public static Dataset generateDataset(String descriptor, String[] data) throws DescriptorException {
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);
    
    // used to convert CATEGORICAL and LABEL attributes to Integer
    List<String>[] values = new List[attrs.length];
    
    int id = 0;
    for (String aData : data) {
      if (aData.isEmpty()) {
        continue;
      }
      
      if (parseString(id, attrs, values, aData) != null) {
        id++;
      }
    }
    
    return new Dataset(attrs, values, id);
  }
  
  /**
   * constructs the data
   * 
   * @param attrs
   *          attributes description
   * @param vectors
   *          data elements
   * @param values
   *          used to convert CATEGORICAL attributes to Integer
   */
  /*
  private static Data constructData(Attribute[] attrs, List<Instance> vectors, List<String>[] values) {
    Dataset dataset = new Dataset(attrs, values, vectors.size());
    
    return new Data(dataset, vectors);
  }
   */
}
