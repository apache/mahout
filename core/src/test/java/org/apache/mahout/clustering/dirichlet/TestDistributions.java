package org.apache.mahout.clustering.dirichlet;

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

import junit.framework.TestCase;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.Vector;

public class TestDistributions extends TestCase {

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    UncommonDistributions.init("Mahout=Hadoop+ML".getBytes());
  }

  public void testRbeta() {
    for (double i = 0.01; i < 20; i += 0.25) {
      System.out.println("rBeta(6,1," + i + ")="
          + UncommonDistributions.rBeta(6, 1, i).asFormatString());
    }
  }

  public void testRchisq() {
    for (int i = 0; i < 50; i++) {
      System.out
          .println("rChisq(" + i + ")=" + UncommonDistributions.rChisq(i));
    }
  }

  public void testRnorm() {
    for (int i = 1; i < 50; i++) {
      System.out.println("rNorm(6,1," + i + ")="
          + UncommonDistributions.rNorm(1, i));
    }
  }

  public void testDnorm() {
    for (int i = -30; i < 30; i++) {
      double d = (i * 0.1);
      double dnorm = UncommonDistributions.dNorm(d, 0, 1);
      byte[] bar = new byte[(int) (dnorm * 100)];
      for (int j = 0; j < bar.length; j++) {
        bar[j] = '*';
      }
      String baz = new String(bar);
      System.out.println(baz);
    }
  }

  public void testDnorm2() {
    for (int i = -30; i < 30; i++) {
      double d = (i * 0.1);
      double dnorm = UncommonDistributions.dNorm(d, 0, 2);
      byte[] bar = new byte[(int) (dnorm * 100)];
      for (int j = 0; j < bar.length; j++) {
        bar[j] = '*';
      }
      String baz = new String(bar);
      System.out.println(baz);
    }
  }

  public void testDnorm1() {
    for (int i = -10; i < 10; i++) {
      double d = (i * 0.1);
      double dnorm = UncommonDistributions.dNorm(d, 0, 0.2);
      byte[] bar = new byte[(int) (dnorm * 100)];
      for (int j = 0; j < bar.length; j++) {
        bar[j] = '*';
      }
      String baz = new String(bar);
      System.out.println(baz);
    }
  }

  public void testRmultinom1() {
    double[] b = {0.4, 0.6};
    Vector v = new DenseVector(b);
    Vector t = v.like();
    for (int i = 1; i <= 100; i++) {
      Vector multinom = UncommonDistributions.rMultinom(100, v);
      t = t.plus(multinom);
    }
    System.out.println("sum(rMultinom(" + 100 + ", [0.4, 0.6]))/100="
        + t.divide(100).asFormatString());

  }

  public void testRmultinom2() {
    double[] b = {0.1, 0.2, 0.7};
    Vector v = new DenseVector(b);
    Vector t = v.like();
    for (int i = 1; i <= 100; i++) {
      Vector multinom = UncommonDistributions.rMultinom(100, v);
      t = t.plus(multinom);
    }
    System.out.println("sum(rMultinom(" + 100 + ", [ 0.1, 0.2, 0.7 ]))/100="
        + t.divide(100).asFormatString());

  }

  public void testRmultinom() {
    double[] b = {0.1, 0.2, 0.8};
    Vector v = new DenseVector(b);
    for (int i = 1; i <= 100; i++) {
      System.out.println("rMultinom(" + 100 + ", [0.1, 0.2, 0.8])="
          + UncommonDistributions.rMultinom(100, v).asFormatString());
    }
  }
}
