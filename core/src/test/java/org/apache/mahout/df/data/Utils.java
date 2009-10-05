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

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.df.callback.PredictionCallback;
import org.apache.mahout.df.data.Dataset.Attribute;
import org.slf4j.Logger;

/**
 * Helper methods used by the tests
 *
 */
public class Utils {
  private Utils() {
  }

  private static class LogCallback implements PredictionCallback {
  
    private final Logger log;
  
    private LogCallback(Logger log) {
      this.log = log;
    }
  
    @Override
    public void prediction(int treeId, int instanceId, int prediction) {
      log.info(String.format("treeId:%04d, instanceId:%06d, prediction:%d",
          treeId, instanceId, prediction));
    }
  
  }

  /** Used when generating random CATEGORICAL values */
  protected static final int CATEGORICAL_RANGE = 100;

  /**
   * Generates a random list of tokens
   * <ul>
   * <li>each attribute has 50% chance to be NUMERICAL ('N') or CATEGORICAL
   * ('C')</li>
   * <li>10% of the attributes are IGNORED ('I')</li>
   * <li>one randomly chosen attribute becomes the LABEL ('L')</li>
   * </ul>
   * 
   * @param rng Random number generator
   * @param nbTokens number of tokens to generate
   * @return
   */
  public static char[] randomTokens(Random rng, int nbTokens) {
    char[] result = new char[nbTokens];

    for (int token = 0; token < nbTokens; token++) {
      double rand = rng.nextDouble();
      if (rand < 0.1)
        result[token] = 'I'; // IGNORED
      else if (rand < 0.5)
        result[token] = 'N'; // NUMERICAL
      else
        result[token] = 'C'; // CATEGORICAL
    }

    // choose the label
    result[rng.nextInt(nbTokens)] = 'L';

    return result;
  }

  /**
   * Generates a space-separated String that contains all the tokens
   * 
   * @param tokens
   * @return
   */
  public static String generateDescriptor(char[] tokens) {
    StringBuilder builder = new StringBuilder();

    for (char token1 : tokens) {
      builder.append(token1).append(' ');
    }

    return builder.toString();
  }

  /**
   * Generates a random descriptor as follows:<br>
   * <ul>
   * <li>each attribute has 50% chance to be NUMERICAL or CATEGORICAL</li>
   * <li>10% of the attributes are IGNORED</li>
   * <li>one randomly chosen attribute becomes the LABEL</li>
   * </ul>
   * 
   * @param nbAttributes
   * @return
   */
  public static String randomDescriptor(Random rng, int nbAttributes) {
    return generateDescriptor(randomTokens(rng, nbAttributes));
  }

  /**
   * generates random data
   * 
   * @param rng Random number generator
   * @param nbAttributes number of attributes
   * @param number of data lines to generate
   * @return
   * @throws Exception 
   */
  public static double[][] randomDoubles(Random rng, int nbAttributes,int number) throws DescriptorException {
    String descriptor = randomDescriptor(rng, nbAttributes);
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);

    double[][] data = new double[number][];

    for (int index = 0; index < number; index++) {
      data[index] = randomVector(rng, attrs);
    }

    return data;
  }

  /**
   * generates random data based on the given descriptor
   * 
   * @param rng Random number generator
   * @param descriptor attributes description
   * @param number number of data lines to generate
   */
  public static double[][] randomDoubles(Random rng, String descriptor, int number) throws DescriptorException {
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);

    double[][] data = new double[number][];

    for (int index = 0; index < number; index++) {
      data[index] = randomVector(rng, attrs);
    }

    return data;
  }

  /**
   * Generates random data
   * 
   * @param rng Random number generator
   * @param nbAttributes number of attributes
   * @param size data size
   * @return
   * @throws Exception 
   */
  public static Data randomData(Random rng, int nbAttributes, int size) throws DescriptorException {
    String descriptor = randomDescriptor(rng, nbAttributes);
    double[][] source = randomDoubles(rng, descriptor, size);
    String[] sData = double2String(source);
    Dataset dataset = DataLoader.generateDataset(descriptor, sData);
    
    return DataLoader.loadData(dataset, sData);
  }

  /**
   * generates a random vector based on the given attributes.<br>
   * the attributes' values are generated as follows :<br>
   * <ul>
   * <li>each IGNORED attribute receives a Double.NaN</li>
   * <li>each NUMERICAL attribute receives a random double</li>
   * <li>each CATEGORICAL and LABEL attribute receives a random integer in the
   * range [0, CATEGORICAL_RANGE[</li>
   * </ul>
   * 
   * @param rng
   * @param attrs attributes description
   * @return
   */
  protected static double[] randomVector(Random rng, Attribute[] attrs) {
    double[] vector = new double[attrs.length];

    for (int attr = 0; attr < attrs.length; attr++) {
      if (attrs[attr].isIgnored())
        vector[attr] = Double.NaN;
      else if (attrs[attr].isNumerical())
        vector[attr] = rng.nextDouble();
      else
        // CATEGORICAL or LABEL
        vector[attr] = rng.nextInt(CATEGORICAL_RANGE);
    }

    return vector;
  }

  /**
   * converts a double array to a comma-separated string
   * 
   * @param v double array
   * @return comma-separated string
   */
  protected static String double2String(double[] v) {
    StringBuilder builder = new StringBuilder();

    for (double aV : v) {
      builder.append(aV).append(',');
    }

    return builder.toString();
  }

  /**
   * converts an array of double arrays to an array of comma-separated strings
   * 
   * @param source array of double arrays
   * @return array of comma-separated strings
   */
  public static String[] double2String(double[][] source) {
    String[] output = new String[source.length];

    for (int index = 0; index < source.length; index++) {
      output[index] = double2String(source[index]);
    }

    return output;
  }

  /**
   * Generates random data with same label value
   * 
   * @param rng
   * @param descriptor
   * @param number data size
   * @param value label value
   * @return
   * @throws Exception 
   */
  public static double[][] randomDoublesWithSameLabel(Random rng,
      String descriptor, int number, int value) throws DescriptorException {
    int label = findLabel(descriptor);
    
    double[][] source = randomDoubles(rng, descriptor, number);
    
    for (int index = 0; index < number; index++) {
      source[index][label] = value;
    }

    return source;
  }

  /**
   * finds the label attribute's index
   * 
   * @param descriptor
   * @return
   * @throws Exception 
   */
  public static int findLabel(String descriptor) throws DescriptorException {
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);
    return ArrayUtils.indexOf(attrs, Attribute.LABEL);
  }

  public static void writeDataToFile(String[] sData, Path path) throws IOException {
    BufferedWriter output = new BufferedWriter(new OutputStreamWriter(
        new FileOutputStream(path.toString()), Charset.forName("UTF-8")));
    try {
      for (String line : sData) {
        output.write(line);
        output.write('\n');
      }
      output.flush();
    } finally {
      output.close();
    }
  
  }

  public static Path writeDataToTestFile(String[] sData) throws IOException {
    Path testData = new Path("testdata/Data");
    FileSystem fs = testData.getFileSystem(new Configuration());
    if (!fs.exists(testData))
      fs.mkdirs(testData);
  
    Path path = new Path(testData, "DataLoaderTest.data");
  
    writeDataToFile(sData, path);
  
    return path;
  }

  public static Path writeDatasetToTestFile(Dataset dataset) throws IOException {
    Path testData = new Path("testdata/Dataset");
    FileSystem fs = testData.getFileSystem(new Configuration());
    if (!fs.exists(testData))
      fs.mkdirs(testData);
  
    Path datasetPath = new Path(testData, "dataset.info");
    FSDataOutputStream out = fs.create(datasetPath);
  
    try {
      dataset.write(out);
    } finally {
      out.close();
    }
  
    return datasetPath;
  }

  /**
   * Split the data into numMaps splits
   * 
   * @param sData
   * @param numMaps
   * @return
   */
  public static String[][] splitData(String[] sData, int numMaps) {
    int nbInstances = sData.length;
    int partitionSize = nbInstances / numMaps;
  
    String[][] splits = new String[numMaps][];
  
    for (int partition = 0; partition < numMaps; partition++) {
      int from = partition * partitionSize;
      int to = (partition == (numMaps - 1)) ? nbInstances : (partition + 1)
          * partitionSize;
  
      splits[partition] = Arrays.copyOfRange(sData, from, to);
    }
  
    return splits;
  }
}
