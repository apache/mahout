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

package org.apache.mahout.classifier.evaluation;

import org.apache.mahout.math.list.DoubleArrayList;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

public class AucTest {
  @Test
  public void testAuc1() {
    // sampled from normal distribution using rnorm(10) in R
    double[] s0 = new double[]{1.14208259646056, 0.486167157248001, 1.42408001542355, -0.227908542284331, 2.72432780169982, 0.494097293912641, -0.152566866170993, -0.418360266395271, 0.359475300232312, 1.35565069667582};
    // sampled using rnorm(15, m=1) in R
    double[] s1 = new double[]{0.505788456871571, 2.38828971158414, 1.59711036686792, 2.05341387608687, 0.0425382591225825, -1.35528802598249, -0.0317145894412825, 1.64431909027163, 0.943089100695804, 0.85580534449119, 0.954319817182506, 1.75469439257183, 1.71974400862853, -0.178732498023014, 0.844112471094082};

    Auc x = new Auc();
    for (double v : s0) {
      x.add(0, v);
    }
    for (double v : s1) {
      x.add(1, v);
    }
    // value verified using ROCR package in R
    assertEquals(0.6133333, x.auc(), 1e-6);
  }

  @Test
  public void testAuc2() {
    // check for behavior with ties
    // s0 gets ranks 1.5, 1.5, 3, 6, 6, 6, 9
    double[] s0 = new double[]{-2, -2, -1, 0, 0, 0, 1};
    // s1 gets ranks 6, 6, 10
    // also, when s1 has highest score, we tickle a special case
    double[] s1 = new double[]{0, 0, 2};

    Auc x = new Auc();
    for (double v : s0) {
      x.add(0, v);
    }
    for (double v : s1) {
      x.add(1, v);
    }
    // value verified using ROCR package in R
    assertEquals(0.7619048, x.auc(), 1e-6);
  }

  @Test
  public void testSetMaxBufferSize() {
    Random rand = new Random(1);
    Auc x1 = new Auc(new Random(2));
    Auc x2 = new Auc(new Random(3));
    x2.setMaxBufferSize(100001);

    // get some normally distributed data
    DoubleArrayList buf0 = new DoubleArrayList();
    DoubleArrayList buf1 = new DoubleArrayList();
    for (int i = 0; i < 100000; i++) {
      buf0.add(rand.nextGaussian());
      buf1.add(rand.nextGaussian() + 1);
    }
    // pre-sort it just to be nasty.  If the sampling is biased
    // this will tend to show it
    buf0.sort();
    buf1.sort();

    for (int i = 0; i < 100000; i++) {
      x1.add(0, buf0.get(i));
      x1.add(1, buf1.get(i));

      x2.add(0, buf0.get(i));
    }

    assertEquals(0.5, x2.auc(), 0.0);

    for (int i = 0; i < 100000; i++) {
      x2.add(1, buf1.get(i));
    }

    double a1 = x1.auc();
    double a2 = x2.auc();
    assertEquals(a1, a2, 0.025);
    assertEquals(0.76, a2, 0.002);
  }
}
