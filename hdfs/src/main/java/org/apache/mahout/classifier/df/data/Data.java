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

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.classifier.df.data.conditions.Condition;

import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * Holds a list of vectors and their corresponding Dataset. contains various operations that deals with the
 * vectors (subset, count,...)
 * 
 */
public class Data implements Cloneable {
  
  private final List<Instance> instances;
  
  private final Dataset dataset;

  public Data(Dataset dataset) {
    this.dataset = dataset;
    this.instances = Lists.newArrayList();
  }

  public Data(Dataset dataset, List<Instance> instances) {
    this.dataset = dataset;
    this.instances = Lists.newArrayList(instances);
  }
  
  /**
   * @return the number of elements
   */
  public int size() {
    return instances.size();
  }
  
  /**
   * @return true if this data contains no element
   */
  public boolean isEmpty() {
    return instances.isEmpty();
  }
  
  /**
   * @param v
   *          element whose presence in this list if to be searched
   * @return true is this data contains the specified element.
   */
  public boolean contains(Instance v) {
    return instances.contains(v);
  }

    /**
   * Returns the element at the specified position
   * 
   * @param index
   *          index of element to return
   * @return the element at the specified position
   * @throws IndexOutOfBoundsException
   *           if the index is out of range
   */
  public Instance get(int index) {
    return instances.get(index);
  }
  
  /**
   * @return the subset from this data that matches the given condition
   */
  public Data subset(Condition condition) {
    List<Instance> subset = Lists.newArrayList();
    
    for (Instance instance : instances) {
      if (condition.isTrueFor(instance)) {
        subset.add(instance);
      }
    }
    
    return new Data(dataset, subset);
  }

    /**
   * if data has N cases, sample N cases at random -but with replacement.
   */
  public Data bagging(Random rng) {
    int datasize = size();
    List<Instance> bag = Lists.newArrayListWithCapacity(datasize);
    
    for (int i = 0; i < datasize; i++) {
      bag.add(instances.get(rng.nextInt(datasize)));
    }
    
    return new Data(dataset, bag);
  }
  
  /**
   * if data has N cases, sample N cases at random -but with replacement.
   * 
   * @param sampled
   *          indicating which instance has been sampled
   * 
   * @return sampled data
   */
  public Data bagging(Random rng, boolean[] sampled) {
    int datasize = size();
    List<Instance> bag = Lists.newArrayListWithCapacity(datasize);
    
    for (int i = 0; i < datasize; i++) {
      int index = rng.nextInt(datasize);
      bag.add(instances.get(index));
      sampled[index] = true;
    }
    
    return new Data(dataset, bag);
  }
  
  /**
   * Splits the data in two, returns one part, and this gets the rest of the data. <b>VERY SLOW!</b>
   */
  public Data rsplit(Random rng, int subsize) {
    List<Instance> subset = Lists.newArrayListWithCapacity(subsize);
    
    for (int i = 0; i < subsize; i++) {
      subset.add(instances.remove(rng.nextInt(instances.size())));
    }
    
    return new Data(dataset, subset);
  }
  
  /**
   * checks if all the vectors have identical attribute values
   * 
   * @return true is all the vectors are identical or the data is empty<br>
   *         false otherwise
   */
  public boolean isIdentical() {
    if (isEmpty()) {
      return true;
    }
    
    Instance instance = get(0);
    for (int attr = 0; attr < dataset.nbAttributes(); attr++) {
      for (int index = 1; index < size(); index++) {
        if (get(index).get(attr) != instance.get(attr)) {
          return false;
        }
      }
    }
    
    return true;
  }
  
  /**
   * checks if all the vectors have identical label values
   */
  public boolean identicalLabel() {
    if (isEmpty()) {
      return true;
    }
    
    double label = dataset.getLabel(get(0));
    for (int index = 1; index < size(); index++) {
      if (dataset.getLabel(get(index)) != label) {
        return false;
      }
    }
    
    return true;
  }
  
  /**
   * finds all distinct values of a given attribute
   */
  public double[] values(int attr) {
    Collection<Double> result = Sets.newHashSet();
    
    for (Instance instance : instances) {
      result.add(instance.get(attr));
    }
    
    double[] values = new double[result.size()];
    
    int index = 0;
    for (Double value : result) {
      values[index++] = value;
    }
    
    return values;
  }
  
  @Override
  public Data clone() {
    return new Data(dataset, Lists.newArrayList(instances));
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof Data)) {
      return false;
    }
    
    Data data = (Data) obj;
    
    return instances.equals(data.instances) && dataset.equals(data.dataset);
  }
  
  @Override
  public int hashCode() {
    return instances.hashCode() + dataset.hashCode();
  }
  
  /**
   * extract the labels of all instances
   */
  public double[] extractLabels() {
    double[] labels = new double[size()];
    
    for (int index = 0; index < labels.length; index++) {
      labels[index] = dataset.getLabel(get(index));
    }
    
    return labels;
  }

    /**
   * finds the majority label, breaking ties randomly<br>
   * This method can be used when the criterion variable is the categorical attribute.
   *
   * @return the majority label value
   */
  public int majorityLabel(Random rng) {
    // count the frequency of each label value
    int[] counts = new int[dataset.nblabels()];
    
    for (int index = 0; index < size(); index++) {
      counts[(int) dataset.getLabel(get(index))]++;
    }
    
    // find the label values that appears the most
    return DataUtils.maxindex(rng, counts);
  }
  
  /**
   * Counts the number of occurrences of each label value<br>
   * This method can be used when the criterion variable is the categorical attribute.
   * 
   * @param counts
   *          will contain the results, supposed to be initialized at 0
   */
  public void countLabels(int[] counts) {
    for (int index = 0; index < size(); index++) {
      counts[(int) dataset.getLabel(get(index))]++;
    }
  }
  
  public Dataset getDataset() {
    return dataset;
  }
}
