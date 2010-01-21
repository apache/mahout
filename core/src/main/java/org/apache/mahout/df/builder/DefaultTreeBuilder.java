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

package org.apache.mahout.df.builder;

import java.util.Random;

import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.Instance;
import org.apache.mahout.df.data.conditions.Condition;
import org.apache.mahout.df.node.CategoricalNode;
import org.apache.mahout.df.node.Leaf;
import org.apache.mahout.df.node.Node;
import org.apache.mahout.df.node.NumericalNode;
import org.apache.mahout.df.split.IgSplit;
import org.apache.mahout.df.split.OptIgSplit;
import org.apache.mahout.df.split.Split;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Builds a Decision Tree <br>
 * Based on the algorithm described in the "Decision Trees" tutorials by Andrew
 * W. Moore, available at:<br>
 * <br>
 * http://www.cs.cmu.edu/~awm/tutorials
 */
public class DefaultTreeBuilder implements TreeBuilder {

  private static final Logger log = LoggerFactory.getLogger(DefaultTreeBuilder.class);

  /** indicates which CATEGORICAL attributes have already been selected in the parent nodes */
  private boolean[] selected;

  /** number of attributes to select randomly at each node */
  private int m = 1;

  /** IgSplit implementation */
  private IgSplit igSplit;

  public DefaultTreeBuilder() {
    igSplit = new OptIgSplit();
  }

  public void setM(int m) {
    this.m = m;
  }

  public void setIgSplit(IgSplit igSplit) {
    this.igSplit = igSplit;
  }

  @Override
  public Node build(Random rng, Data data) {

    if (selected == null) {
      selected = new boolean[data.getDataset().nbAttributes()];
    }

    if (data.isEmpty())
      return new Leaf(-1);
    if (isIdentical(data))
      return new Leaf(data.majorityLabel(rng));
    if (data.identicalLabel())
      return new Leaf(data.get(0).label);

    int[] attributes = randomAttributes(rng, selected, m);

    // find the best split
    Split best = null;
    for (int attr : attributes) {
      Split split = igSplit.computeSplit(data, attr);
      if (best == null || best.ig < split.ig)
        best = split;
    }

    boolean alreadySelected = selected[best.attr];

    if (alreadySelected) {
      // attribute already selected
      log.warn("attribute {} already selected in a parent node", best.attr);
    }

    Node childNode = null;
    if (data.getDataset().isNumerical(best.attr)) {
      Data loSubset = data.subset(Condition.lesser(best.attr, best.split));
      Node loChild = build(rng, loSubset);

      Data hiSubset = data.subset(Condition.greaterOrEquals(best.attr,
          best.split));
      Node hiChild = build(rng, hiSubset);

      childNode = new NumericalNode(best.attr, best.split, loChild, hiChild);
    } else { // CATEGORICAL attribute
      selected[best.attr] = true;
      
      double[] values = data.values(best.attr);
      Node[] childs = new Node[values.length];

      for (int index = 0; index < values.length; index++) {
        Data subset = data.subset(Condition.equals(best.attr, values[index]));
        childs[index] = build(rng, subset);
      }

      childNode = new CategoricalNode(best.attr, values, childs);

      if (!alreadySelected) {
        selected[best.attr] = false;
      }
    }

    return childNode;
  }

  /**
   * checks if all the vectors have identical attribute values. Ignore selected attributes.
   *
   * @return true is all the vectors are identical or the data is empty<br>
   *         false otherwise
   */
  private boolean isIdentical(Data data) {
    if (data.isEmpty()) return true;

    Instance instance = data.get(0);
    for (int attr = 0; attr < selected.length; attr++) {
    if (selected[attr]) continue;

    for (int index = 1; index < data.size(); index++) {
      if (data.get(index).get(attr) != instance.get(attr))
        return false;
      }
    }

    return true;
  }

  /**
   * Randomly selects m attributes to consider for split, excludes IGNORED and
   * LABEL attributes
   * 
   * @param rng random-numbers generator
   * @param selected attributes' state (selected or not)
   * @param m number of attributes to choose
   * @return
   */
  protected static int[] randomAttributes(Random rng, boolean[] selected, int m) {
    int nbNonSelected = 0; // number of non selected attributes
    for (boolean sel : selected) {
      if (!sel) nbNonSelected++;
    }

    if (nbNonSelected == 0) {
      log.warn("All attributes are selected !");
    }

    int[] result;
    if (nbNonSelected <= m) {
      // return all non selected attributes
      result = new int[nbNonSelected];
      int index = 0;
      for (int attr = 0; attr < selected.length; attr++) {
        if (!selected[attr]) result[index++] = attr;
      }
    } else {
      result = new int[m];
      for (int index = 0; index < m; index++) {
        // randomly choose a "non selected" attribute
        int rind;
        do {
          rind = rng.nextInt(selected.length);
        } while (selected[rind]);

        result[index] = rind;
        selected[rind] = true; // temporarely set the choosen attribute to be selected
      }

      // the choosen attributes are not yet selected
      for (int attr : result) {
        selected[attr] = false;
      }
    }

    return result;
  }
}
