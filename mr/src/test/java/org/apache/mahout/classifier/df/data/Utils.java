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

package org.apache.mahout.classifier.df.data;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.df.data.Dataset.Attribute;
import org.apache.mahout.common.MahoutTestCase;

/**
 * Helper methods used by the tests
 *
 */
@Deprecated
public final class Utils {

  private Utils() {}

  /** Used when generating random CATEGORICAL values */
  private static final int CATEGORICAL_RANGE = 100;

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
   */
  public static char[] randomTokens(Random rng, int nbTokens) {
    char[] result = new char[nbTokens];

    for (int token = 0; token < nbTokens; token++) {
      double rand = rng.nextDouble();
      if (rand < 0.1) {
        result[token] = 'I'; // IGNORED
      } else if (rand >= 0.5) {
        result[token] = 'C';
      } else {
        result[token] = 'N'; // NUMERICAL
      } // CATEGORICAL
    }

    // choose the label
    result[rng.nextInt(nbTokens)] = 'L';

    return result;
  }

  /**
   * Generates a space-separated String that contains all the tokens
   */
  public static String generateDescriptor(char[] tokens) {
    StringBuilder builder = new StringBuilder();

    for (char token : tokens) {
      builder.append(token).append(' ');
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
   */
  public static String randomDescriptor(Random rng, int nbAttributes) {
    return generateDescriptor(randomTokens(rng, nbAttributes));
  }

  /**
   * generates random data based on the given descriptor
   * 
   * @param rng Random number generator
   * @param descriptor attributes description
   * @param number number of data lines to generate
   */
  public static double[][] randomDoubles(Random rng, CharSequence descriptor, boolean regression, int number)
    throws DescriptorException {
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);

    double[][] data = new double[number][];

    for (int index = 0; index < number; index++) {
      data[index] = randomVector(rng, attrs, regression);
    }

    return data;
  }

  /**
   * Generates random data
   * 
   * @param rng Random number generator
   * @param nbAttributes number of attributes
   * @param regression true is the label should be numerical
   * @param size data size
   */
  public static Data randomData(Random rng, int nbAttributes, boolean regression, int size) throws DescriptorException {
    String descriptor = randomDescriptor(rng, nbAttributes);
    double[][] source = randomDoubles(rng, descriptor, regression, size);
    String[] sData = double2String(source);
    Dataset dataset = DataLoader.generateDataset(descriptor, regression, sData);
    
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
   * @param attrs attributes description
   */
  private static double[] randomVector(Random rng, Attribute[] attrs, boolean regression) {
    double[] vector = new double[attrs.length];

    for (int attr = 0; attr < attrs.length; attr++) {
      if (attrs[attr].isIgnored()) {
        vector[attr] = Double.NaN;
      } else if (attrs[attr].isNumerical()) {
        vector[attr] = rng.nextDouble();
      } else if (attrs[attr].isCategorical()) {
        vector[attr] = rng.nextInt(CATEGORICAL_RANGE);
      } else { // LABEL
      	if (regression) {
          vector[attr] = rng.nextDouble();
      	} else {
          vector[attr] = rng.nextInt(CATEGORICAL_RANGE);
      	}
      }
    }

    return vector;
  }

  /**
   * converts a double array to a comma-separated string
   * 
   * @param v double array
   * @return comma-separated string
   */
  private static String double2String(double[] v) {
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
   * @param number data size
   * @param value label value
   */
  public static double[][] randomDoublesWithSameLabel(Random rng,
                                                      CharSequence descriptor,
                                                      boolean regression,
                                                      int number,
                                                      int value) throws DescriptorException {
    int label = findLabel(descriptor);
    
    double[][] source = randomDoubles(rng, descriptor, regression, number);
    
    for (int index = 0; index < number; index++) {
      source[index][label] = value;
    }

    return source;
  }

  /**
   * finds the label attribute's index
   */
  public static int findLabel(CharSequence descriptor) throws DescriptorException {
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);
    return ArrayUtils.indexOf(attrs, Attribute.LABEL);
  }

  private static void writeDataToFile(String[] sData, Path path) throws IOException {
    BufferedWriter output = null;
    try {
      output = Files.newWriter(new File(path.toString()), Charsets.UTF_8);
      for (String line : sData) {
        output.write(line);
        output.write('\n');
      }
    } finally {
      Closeables.close(output, false);
    }
  
  }

  public static Path writeDataToTestFile(String[] sData) throws IOException {
    Path testData = new Path("testdata/Data");
    MahoutTestCase ca = new MahoutTestCase();
    FileSystem fs = testData.getFileSystem(ca.getConfiguration());
    if (!fs.exists(testData)) {
      fs.mkdirs(testData);
    }
  
    Path path = new Path(testData, "DataLoaderTest.data");
  
    writeDataToFile(sData, path);
  
    return path;
  }

  /**
   * Split the data into numMaps splits
   */
  public static String[][] splitData(String[] sData, int numMaps) {
    int nbInstances = sData.length;
    int partitionSize = nbInstances / numMaps;
  
    String[][] splits = new String[numMaps][];
  
    for (int partition = 0; partition < numMaps; partition++) {
      int from = partition * partitionSize;
      int to = partition == (numMaps - 1) ? nbInstances : (partition + 1) * partitionSize;
  
      splits[partition] = Arrays.copyOfRange(sData, from, to);
    }
  
    return splits;
  }
}
