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

package org.apache.mahout.matrix;

import junit.framework.TestCase;

public class VectorTest extends TestCase {

  public VectorTest(String s) {
    super(s);
  }

  public void testSparseVector() throws Exception {
    SparseVector vec1 = new SparseVector(3);
    SparseVector vec2 = new SparseVector(3);
    doTestVectors(vec1, vec2);
  }

  private static void doTestVectors(Vector left, Vector right) {
    left.setQuick(0, 1);
    left.setQuick(1, 2);
    left.setQuick(2, 3);
    right.setQuick(0, 4);
    right.setQuick(1, 5);
    right.setQuick(2, 6);
    double result = left.dot(right);
    assertEquals(result + " does not equal: " + 32, 32.0, result);
  }


  public void testDenseVector() throws Exception {
    DenseVector vec1 = new DenseVector(3);
    DenseVector vec2 = new DenseVector(3);
    doTestVectors(vec1, vec2);

  }

  public void testVectorView() throws Exception {
    SparseVector vec1 = new SparseVector(3);
    SparseVector vec2 = new SparseVector(6);
    VectorView vecV1 = new VectorView(vec1, 0, 3);
    VectorView vecV2 = new VectorView(vec2, 2, 3);
    doTestVectors(vecV1, vecV2);
  }

  /**
   * Asserts a vector using enumeration equals a given dense vector
   */
  private static void doTestEnumeration(double[] apriori, Vector vector) {
    double[] test = new double[apriori.length];
    for (Vector.Element e : vector) {
      test[e.index()] = e.get();
    }

    for (int i = 0; i<test.length; i++) {
      assertEquals(apriori[i], test[i]);
    }
  }

  public void testEnumeration() throws Exception {
    double[] apriori = {0, 1, 2, 3, 4};

    doTestEnumeration(apriori, new VectorView(new DenseVector(new double[]{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), 2, 5));
    
    doTestEnumeration(apriori, new DenseVector(new double[]{0, 1, 2, 3, 4}));

    SparseVector sparse = new SparseVector(5);
    sparse.set(0, 0);
    sparse.set(1, 1);
    sparse.set(2, 2);
    sparse.set(3, 3);
    sparse.set(4, 4);
    doTestEnumeration(apriori, sparse);
  }

}