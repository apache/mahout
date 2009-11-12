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

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.lang.ArrayUtils;
import org.apache.mahout.df.data.Data;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.data.conditions.Condition;
import org.apache.mahout.df.node.CategoricalNode;
import org.apache.mahout.df.node.Leaf;
import org.apache.mahout.df.node.Node;
import org.apache.mahout.df.node.NumericalNode;
import org.apache.mahout.df.split.IgSplit;
import org.apache.mahout.df.split.OptIgSplit;
import org.apache.mahout.df.split.Split;

/**
 * Builds a Decision Tree <br>
 * Based on the algorithm described in the "Decision Trees" tutorials by Andrew
 * W. Moore, available at:<br>
 * <br>
 * http://www.cs.cmu.edu/~awm/tutorials
 */
public class DefaultTreeBuilder implements TreeBuilder {

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

    if (data.isEmpty())
      return new Leaf(-1);
    if (data.isIdentical())
      return new Leaf(data.majorityLabel(rng));
    if (data.identicalLabel())
      return new Leaf(data.get(0).label);

    int[] attributes = randomAttributes(data.getDataset(), rng, m);

    // find the best split
    Split best = null;
    for (int attr : attributes) {
      Split split = igSplit.computeSplit(data, attr);
      if (best == null || best.ig < split.ig)
        best = split;
    }

    if (data.getDataset().isNumerical(best.attr)) {
      Data loSubset = data.subset(Condition.lesser(best.attr, best.split));
      Node loChild = build(rng, loSubset);

      Data hiSubset = data.subset(Condition.greaterOrEquals(best.attr,
          best.split));
      Node hiChild = build(rng, hiSubset);

      return new NumericalNode(best.attr, best.split, loChild, hiChild);
    } else { // CATEGORICAL attribute
      double[] values = data.values(best.attr);
      Node[] childs = new Node[values.length];

      for (int index = 0; index < values.length; index++) {
        Data subset = data.subset(Condition.equals(best.attr, values[index]));
        childs[index] = build(rng, subset);
      }

      return new CategoricalNode(best.attr, values, childs);
    }
  }

  /**
   * Randomly selects m attributes to consider for split, excludes IGNORED and
   * LABEL attributes
   * 
   * @param dataset
   * @param rng
   * @param m number of attributes to select
   * @return
   */
  protected static int[] randomAttributes(Dataset dataset, Random rng, int m) {
    if (m > dataset.nbAttributes()) {
      throw new IllegalArgumentException("m > num attributes");
    }

    int[] result = new int[m];

    Arrays.fill(result, -1);

    for (int index = 0; index < m; index++) {
      int rvalue;
      do {
        rvalue = rng.nextInt(dataset.nbAttributes());
      } while (ArrayUtils.contains(result, rvalue));

      result[index] = rvalue;
    }

    return result;
  }
}
