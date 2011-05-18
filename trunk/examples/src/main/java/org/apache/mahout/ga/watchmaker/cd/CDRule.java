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

import java.util.Random;

import com.google.common.base.Preconditions;
import org.uncommons.maths.binary.BitString;

/**
 * Binary classification rule of the form:
 *
 * <pre>
 * if (condition1 &amp;&amp; condition2 &amp;&amp; ... ) then
 *   class = 1
 * else
 *   class = 0
 * </pre>
 *
 * where conditioni = (wi): attributi oi vi <br>
 * <ul>
 * <li>wi is the weight of the condition: <br>
 * {@code if (wi < a given threshold) then conditioni is not taken into consideration}
 * </li>
 * <li>oi is an operator ('&lt;' or '&gt;=')</li>
 * </ul>
 */
public final class CDRule implements Rule {
  
  private final double threshold;
  private final int nbConditions;
  private final double[] weights;
  private final BitString operators;
  private final double[] values;
  
  /**
   * @param threshold
   *          condition activation threshold
   */
  public CDRule(double threshold) {
    // crossover needs at least 2 attributes
    Preconditions.checkArgument(threshold >= 0.0 && threshold <= 1.0, "Threshold must be in [0,1]");

    this.threshold = threshold;
    
    // the label is not included in the conditions
    this.nbConditions = DataSet.getDataSet().getNbAttributes() - 1;
    
    weights = new double[nbConditions];
    operators = new BitString(nbConditions);
    values = new double[nbConditions];
  }
  
  /**
   * Random rule.
   */
  public CDRule(double threshold, Random rng) {
    this(threshold);
    
    DataSet dataset = DataSet.getDataSet();
    
    for (int condInd = 0; condInd < nbConditions; condInd++) {
      int attrInd = attributeIndex(condInd);
      
      setW(condInd, rng.nextDouble());
      setO(condInd, rng.nextBoolean());
      if (dataset.isNumerical(attrInd)) {
        setV(condInd, randomNumerical(dataset, attrInd, rng));
      } else {
        setV(condInd, randomCategorical(dataset, attrInd, rng));
      }
    }
  }

  /**
   * Copy Constructor
   */
  public CDRule(CDRule ind) {
    threshold = ind.threshold;
    nbConditions = ind.nbConditions;

    weights = ind.weights.clone();
    operators = ind.operators.clone();
    values = ind.values.clone();
  }
  
  private static double randomNumerical(DataSet dataset, int attrInd, Random rng) {
    double max = dataset.getMax(attrInd);
    double min = dataset.getMin(attrInd);
    return rng.nextDouble() * (max - min) + min;
  }
  
  private static double randomCategorical(DataSet dataset, int attrInd, Random rng) {
    int nbcategories = dataset.getNbValues(attrInd);
    return rng.nextInt(nbcategories);
  }

  /**
   * if all the active conditions are met returns 1, else returns 0.
   */
  @Override
  public int classify(DataLine dl) {
    for (int condInd = 0; condInd < nbConditions; condInd++) {
      if (!condition(condInd, dl)) {
        return 0;
      }
    }
    return 1;
  }
  
  /**
   * Makes sure that the label is not handled by any condition.
   * 
   * @param condInd
   *          condition index
   * @return attribute index
   */
  public static int attributeIndex(int condInd) {
    int labelpos = DataSet.getDataSet().getLabelIndex();
    return condInd < labelpos ? condInd : condInd + 1;
  }
  
  /**
   * Returns the value of the condition.
   * 
   * @param condInd
   *          index of the condition
   */
  boolean condition(int condInd, DataLine dl) {
    int attrInd = attributeIndex(condInd);
    
    // is the condition active
    if (getW(condInd) < threshold) {
      return true; // no
    }

    return DataSet.getDataSet().isNumerical(attrInd)
        ? numericalCondition(condInd, dl)
        : categoricalCondition(condInd, dl);
  }
  
  boolean numericalCondition(int condInd, DataLine dl) {
    int attrInd = attributeIndex(condInd);

    return getO(condInd) ? dl.getAttribute(attrInd) >= getV(condInd) : dl.getAttribute(attrInd) < getV(condInd);
  }
  
  boolean categoricalCondition(int condInd, DataLine dl) {
    int attrInd = attributeIndex(condInd);

    return getO(condInd) ? dl.getAttribute(attrInd) == getV(condInd) : dl.getAttribute(attrInd) != getV(condInd);
  }
  
  @Override
  public String toString() {
    StringBuilder buffer = new StringBuilder();
    
    buffer.append("CDRule = [");
    boolean empty = true;
    for (int condInd = 0; condInd < nbConditions; condInd++) {
      if (getW(condInd) >= threshold) {
        if (!empty) {
          buffer.append(" && ");
        }
        
        buffer.append("attr").append(attributeIndex(condInd)).append(' ').append(
          getO(condInd) ? ">=" : "<");
        buffer.append(' ').append(getV(condInd));
        
        empty = false;
      }
    }
    buffer.append(']');
    
    return buffer.toString();
  }
  
  public int getNbConditions() {
    return nbConditions;
  }
  
  public double getW(int index) {
    return weights[index];
  }
  
  public void setW(int index, double w) {
    weights[index] = w;
  }
  
  /**
   * operator
   * 
   * @return true if '&gt;='; false if '&lt;'
   */
  public boolean getO(int index) {
    return operators.getBit(index);
  }
  
  /**
   * set the operator
   * 
   * @param o true if '&gt;='; false if '&lt;'
   */
  public void setO(int index, boolean o) {
    operators.setBit(index, o);
  }
  
  public double getV(int index) {
    return values[index];
  }
  
  public void setV(int index, double v) {
    values[index] = v;
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof CDRule)) {
      return false;
    }
    CDRule rule = (CDRule) obj;
    
    for (int index = 0; index < nbConditions; index++) {
      if (!areGenesEqual(this, rule, index)) {
        return false;
      }
    }
    
    return true;
  }
  
  @Override
  public int hashCode() {
    int value = 0;
    for (int index = 0; index < nbConditions; index++) {
      value *= 31;
      value += Double.doubleToLongBits(getW(index)) + (getO(index) ? 1 : 0) + getV(index);
    }
    return value;
  }
  
  /**
   * Compares a given gene between two rules
   *
   * @param index
   *          gene index
   * @return true if the gene is the same
   */
  public static boolean areGenesEqual(CDRule rule1, CDRule rule2, int index) {
    return rule1.getW(index) == rule2.getW(index) && rule1.getO(index) == rule2.getO(index)
           && rule1.getV(index) == rule2.getV(index);
  }
  
  /**
   * Compares two genes from this Rule
   * 
   * @param index1
   *          first gene index
   * @param index2
   *          second gene index
   * @return if the genes are equal
   */
  public boolean areGenesEqual(int index1, int index2) {
    return getW(index1) == getW(index2) && getO(index1) == getO(index2) && getV(index1) == getV(index2);
  }
}
