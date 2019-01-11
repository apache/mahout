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

package org.apache.mahout.math.random;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Multiset;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.QRDecomposition;
import org.junit.Test;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public final class ChineseRestaurantTest extends MahoutTestCase {

  @Test
  public void testDepth() {
    List<Integer> totals = Lists.newArrayList();
    for (int i = 0; i < 1000; i++) {
      ChineseRestaurant x = new ChineseRestaurant(10);
      Multiset<Integer> counts = HashMultiset.create();
      for (int j = 0; j < 100; j++) {
        counts.add(x.sample());
      }
      List<Integer> tmp = Lists.newArrayList();
      for (Integer k : counts.elementSet()) {
        tmp.add(counts.count(k));
      }
      Collections.sort(tmp, Collections.reverseOrder());
      while (totals.size() < tmp.size()) {
        totals.add(0);
      }
      int j = 0;
      for (Integer k : tmp) {
        totals.set(j, totals.get(j) + k);
        j++;
      }
    }

    // these are empirically derived values, not principled ones
    assertEquals(25000.0, (double) totals.get(0), 1000);
    assertEquals(24000.0, (double) totals.get(1), 1000);
    assertEquals(8000.0, (double) totals.get(2), 200);
    assertEquals(1000.0, (double) totals.get(15), 50);
    assertEquals(1000.0, (double) totals.get(20), 40);
  }

  @Test
  public void testExtremeDiscount() {
    ChineseRestaurant x = new ChineseRestaurant(100, 1);
    Multiset<Integer> counts = HashMultiset.create();
    for (int i = 0; i < 10000; i++) {
      counts.add(x.sample());
    }
    assertEquals(10000, x.size());
    for (int i = 0; i < 10000; i++) {
      assertEquals(1, x.count(i));
    }
  }

  @Test
  public void testGrowth() {
    ChineseRestaurant s0 = new ChineseRestaurant(10, 0.0);
    ChineseRestaurant s5 = new ChineseRestaurant(10, 0.5);
    ChineseRestaurant s9 = new ChineseRestaurant(10, 0.9);
    Set<Double> splits = ImmutableSet.of(1.0, 1.5, 2.0, 3.0, 5.0, 8.0);

    double offset0 = 0;
    int k = 0;
    int i = 0;
    Matrix m5 = new DenseMatrix(20, 3);
    Matrix m9 = new DenseMatrix(20, 3);
    while (i <= 200000) {
      double n = i / Math.pow(10, Math.floor(Math.log10(i)));
      if (splits.contains(n)) {
        //System.out.printf("%d\t%d\t%d\t%d\n", i, s0.size(), s5.size(), s9.size());
        if (i > 900) {
          double predict5 = predictSize(m5.viewPart(0, k, 0, 3), i, 0.5);
          assertEquals(predict5, Math.log(s5.size()), 1);

          double predict9 = predictSize(m9.viewPart(0, k, 0, 3), i, 0.9);
          assertEquals(predict9, Math.log(s9.size()), 1);

          //assertEquals(10.5 * Math.log(i) - offset0, s0.size(), 10);
        } else if (i > 50) {
          double x = 10.5 * Math.log(i) - s0.size();
          m5.viewRow(k).assign(new double[]{Math.log(s5.size()), Math.log(i), 1});
          m9.viewRow(k).assign(new double[]{Math.log(s9.size()), Math.log(i), 1});

          k++;
          offset0 += (x - offset0) / k;
        }
        if (i > 10000) {
          assertEquals(0.0, (double) hapaxCount(s0) / s0.size(), 0.25);
          assertEquals(0.5, (double) hapaxCount(s5) / s5.size(), 0.1);
          assertEquals(0.9, (double) hapaxCount(s9) / s9.size(), 0.05);
        }
      }
      s0.sample();
      s5.sample();
      s9.sample();
      i++;
    }
  }

  /**
   * Predict the power law growth in number of unique samples from the first few data points.
   * Also check that the fitted growth coefficient is about right.
   *
   * @param m
   * @param currentIndex        Total data points seen so far.  Unique values should be log(currentIndex)*expectedCoefficient + offset.
   * @param expectedCoefficient What slope do we expect.
   * @return The predicted value for log(currentIndex)
   */
  private static double predictSize(Matrix m, int currentIndex, double expectedCoefficient) {
    int rows = m.rowSize();
    Matrix a = m.viewPart(0, rows, 1, 2);
    Matrix b = m.viewPart(0, rows, 0, 1);

    Matrix ata = a.transpose().times(a);
    Matrix atb = a.transpose().times(b);
    QRDecomposition s = new QRDecomposition(ata);
    Matrix r = s.solve(atb).transpose();
    assertEquals(expectedCoefficient, r.get(0, 0), 0.2);
    return r.times(new DenseVector(new double[]{Math.log(currentIndex), 1})).get(0);
  }

  private static int hapaxCount(ChineseRestaurant s) {
    int r = 0;
    for (int i = 0; i < s.size(); i++) {
      if (s.count(i) == 1) {
        r++;
      }
    }
    return r;
  }
}
