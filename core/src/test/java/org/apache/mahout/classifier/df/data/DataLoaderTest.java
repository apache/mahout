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

import java.util.Collection;
import java.util.Random;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.data.Dataset.Attribute;
import org.junit.Test;

public final class DataLoaderTest extends MahoutTestCase {

  private Random rng;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    rng = RandomUtils.getRandom();
  }

  @Test
  public void testLoadDataWithDescriptor() throws Exception {
    int nbAttributes = 10;
    int datasize = 100;

    // prepare the descriptors
    String descriptor = Utils.randomDescriptor(rng, nbAttributes);
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);

    // prepare the data
    double[][] data = Utils.randomDoubles(rng, descriptor, false, datasize);
    Collection<Integer> missings = Lists.newArrayList();
    String[] sData = prepareData(data, attrs, missings);
    Dataset dataset = DataLoader.generateDataset(descriptor, false, sData);
    Data loaded = DataLoader.loadData(dataset, sData);

    testLoadedData(data, attrs, missings, loaded);
    testLoadedDataset(data, attrs, missings, loaded);

    // regression
    data = Utils.randomDoubles(rng, descriptor, true, datasize);
    missings = Lists.newArrayList();
    sData = prepareData(data, attrs, missings);
    dataset = DataLoader.generateDataset(descriptor, true, sData);
    loaded = DataLoader.loadData(dataset, sData);

    testLoadedData(data, attrs, missings, loaded);
    testLoadedDataset(data, attrs, missings, loaded);
  }

  /**
   * Test method for
   * {@link DataLoader#generateDataset(CharSequence, boolean, String[])}
   */
  @Test
  public void testGenerateDataset() throws Exception {
    int nbAttributes = 10;
    int datasize = 100;

    // prepare the descriptors
    String descriptor = Utils.randomDescriptor(rng, nbAttributes);
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);

    // prepare the data
    double[][] data = Utils.randomDoubles(rng, descriptor, false, datasize);
    Collection<Integer> missings = Lists.newArrayList();
    String[] sData = prepareData(data, attrs, missings);
    Dataset expected = DataLoader.generateDataset(descriptor, false, sData);

    Dataset dataset = DataLoader.generateDataset(descriptor, false, sData);
    
    assertEquals(expected, dataset);

    // regression
    data = Utils.randomDoubles(rng, descriptor, true, datasize);
    missings = Lists.newArrayList();
    sData = prepareData(data, attrs, missings);
    expected = DataLoader.generateDataset(descriptor, true, sData);

    dataset = DataLoader.generateDataset(descriptor, true, sData);
    
    assertEquals(expected, dataset);
}

  /**
   * Converts the data to an array of comma-separated strings and adds some
   * missing values in all but IGNORED attributes
   *
   * @param missings indexes of vectors with missing values
   */
  private String[] prepareData(double[][] data, Attribute[] attrs, Collection<Integer> missings) {
    int nbAttributes = attrs.length;

    String[] sData = new String[data.length];

    for (int index = 0; index < data.length; index++) {
      int missingAttr;
      if (rng.nextDouble() < 0.0) {
        // add a missing value
        missings.add(index);

        // choose a random attribute (not IGNORED)
        do {
          missingAttr = rng.nextInt(nbAttributes);
        } while (attrs[missingAttr].isIgnored());
      } else {
        missingAttr = -1;
      }

      StringBuilder builder = new StringBuilder();

      for (int attr = 0; attr < nbAttributes; attr++) {
        if (attr == missingAttr) {
          // add a missing value here
          builder.append('?').append(',');
        } else {
          builder.append(data[index][attr]).append(',');
        }
      }

      sData[index] = builder.toString();
    }

    return sData;
  }

  /**
   * Test if the loaded data matches the source data
   *
   * @param missings indexes of instance with missing values
   */
  static void testLoadedData(double[][] data, Attribute[] attrs, Collection<Integer> missings, Data loaded) {
    int nbAttributes = attrs.length;

    // check the vectors
    assertEquals("number of instance", data.length - missings.size(), loaded .size());

    // make sure that the attributes are loaded correctly
    int lind = 0;
    for (int index = 0; index < data.length; index++) {
      if (missings.contains(index)) {
        continue;
      }// this vector won't be loaded

      double[] vector = data[index];
      Instance instance = loaded.get(lind);

      int aId = 0;
      for (int attr = 0; attr < nbAttributes; attr++) {
        if (attrs[attr].isIgnored()) {
          continue;
        }

        if (attrs[attr].isNumerical()) {
          assertEquals(vector[attr], instance.get(aId), EPSILON);
          aId++;
        } else if (attrs[attr].isCategorical()) {
          checkCategorical(data, missings, loaded, attr, aId, vector[attr],
              instance.get(aId));
          aId++;
        } else if (attrs[attr].isLabel()) {
          if (loaded.getDataset().isNumerical(aId)) {
            assertEquals(vector[attr], instance.get(aId), EPSILON);
          } else {
            checkCategorical(data, missings, loaded, attr, aId, vector[attr],
              instance.get(aId));
          }
          aId++;
        }
      }
      
      lind++;
    }

  }
  
  /**
   * Test if the loaded dataset matches the source data
   *
   * @param missings indexes of instance with missing values
   */
  static void testLoadedDataset(double[][] data,
                                Attribute[] attrs,
                                Collection<Integer> missings,
                                Data loaded) {
    int nbAttributes = attrs.length;

    int iId = 0;
    for (int index = 0; index < data.length; index++) {
      if (missings.contains(index)) {
        continue;
      }
      
      Instance instance = loaded.get(iId++);

      int aId = 0;
      for (int attr = 0; attr < nbAttributes; attr++) {
        if (attrs[attr].isIgnored()) {
          continue;
        }

        if (attrs[attr].isLabel()) {
          if (!loaded.getDataset().isNumerical(aId)) {
            double nValue = instance.get(aId);
            String oValue = Double.toString(data[index][attr]);
            assertEquals(loaded.getDataset().valueOf(aId, oValue), nValue, EPSILON);
          }
        } else {
          assertEquals(attrs[attr].isNumerical(), loaded.getDataset().isNumerical(aId));
          
          if (attrs[attr].isCategorical()) {
            double nValue = instance.get(aId);
            String oValue = Double.toString(data[index][attr]);
            assertEquals(loaded.getDataset().valueOf(aId, oValue), nValue, EPSILON);
          }
        }
        aId++;
      }
    }

  }

  @Test
  public void testLoadDataFromFile() throws Exception {
    int nbAttributes = 10;
    int datasize = 100;

    // prepare the descriptors
    String descriptor = Utils.randomDescriptor(rng, nbAttributes);
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);

    // prepare the data
    double[][] source = Utils.randomDoubles(rng, descriptor, false, datasize);
    Collection<Integer> missings = Lists.newArrayList();
    String[] sData = prepareData(source, attrs, missings);
    Dataset dataset = DataLoader.generateDataset(descriptor, false, sData);

    Path dataPath = Utils.writeDataToTestFile(sData);
    FileSystem fs = dataPath.getFileSystem(getConfiguration());
    Data loaded = DataLoader.loadData(dataset, fs, dataPath);

    testLoadedData(source, attrs, missings, loaded);

    // regression
    source = Utils.randomDoubles(rng, descriptor, true, datasize);
    missings = Lists.newArrayList();
    sData = prepareData(source, attrs, missings);
    dataset = DataLoader.generateDataset(descriptor, true, sData);

    dataPath = Utils.writeDataToTestFile(sData);
    fs = dataPath.getFileSystem(getConfiguration());
    loaded = DataLoader.loadData(dataset, fs, dataPath);

    testLoadedData(source, attrs, missings, loaded);
}

  /**
   * Test method for
   * {@link DataLoader#generateDataset(CharSequence, boolean, FileSystem, Path)}
   */
  @Test
  public void testGenerateDatasetFromFile() throws Exception {
    int nbAttributes = 10;
    int datasize = 100;

    // prepare the descriptors
    String descriptor = Utils.randomDescriptor(rng, nbAttributes);
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);

    // prepare the data
    double[][] source = Utils.randomDoubles(rng, descriptor, false, datasize);
    Collection<Integer> missings = Lists.newArrayList();
    String[] sData = prepareData(source, attrs, missings);
    Dataset expected = DataLoader.generateDataset(descriptor, false, sData);

    Path path = Utils.writeDataToTestFile(sData);
    FileSystem fs = path.getFileSystem(getConfiguration());
    
    Dataset dataset = DataLoader.generateDataset(descriptor, false, fs, path);
    
    assertEquals(expected, dataset);

    // regression
    source = Utils.randomDoubles(rng, descriptor, false, datasize);
    missings = Lists.newArrayList();
    sData = prepareData(source, attrs, missings);
    expected = DataLoader.generateDataset(descriptor, false, sData);

    path = Utils.writeDataToTestFile(sData);
    fs = path.getFileSystem(getConfiguration());
    
    dataset = DataLoader.generateDataset(descriptor, false, fs, path);
    
    assertEquals(expected, dataset);
  }

  /**
   * each time oValue appears in data for the attribute 'attr', the nValue must
   * appear in vectors for the same attribute.
   *
   * @param attr attribute's index in source
   * @param aId attribute's index in loaded
   * @param oValue old value in source
   * @param nValue new value in loaded
   */
  static void checkCategorical(double[][] source,
                               Collection<Integer> missings,
                               Data loaded,
                               int attr,
                               int aId,
                               double oValue,
                               double nValue) {
    int lind = 0;

    for (int index = 0; index < source.length; index++) {
      if (missings.contains(index)) {
        continue;
      }

      if (source[index][attr] == oValue) {
        assertEquals(nValue, loaded.get(lind).get(aId), EPSILON);
      } else {
        assertFalse(nValue == loaded.get(lind).get(aId));
      }

      lind++;
    }
  }
}
