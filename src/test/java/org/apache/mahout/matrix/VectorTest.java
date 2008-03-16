package org.apache.mahout.matrix;

/**
 * Copyright 2004 The Apache Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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

public class VectorTest extends TestCase {


  public VectorTest(String s) {
    super(s);
  }

  protected void setUp() {
  }

  protected void tearDown() {

  }

  public void testSparseVector() throws Exception {
    SparseVector vec1 = new SparseVector(3);
    SparseVector vec2 = new SparseVector(3);
    testVectors(vec1, vec2);

  }

  private void testVectors(Vector left, Vector right) throws CardinalityException, IndexException {
    left.setQuick(0, 1);
    left.setQuick(1, 2);
    left.setQuick(2, 3);
    right.setQuick(0, 4);
    right.setQuick(1, 5);
    right.setQuick(2, 6);
    double result = left.dot(right);
    assertTrue(result + " does not equal: " + 32, result == 32);
  }


  public void testDenseVector() throws Exception {
    DenseVector vec1 = new DenseVector(3);
    DenseVector vec2 = new DenseVector(3);
    testVectors(vec1, vec2);

  }

  public void testVectorView() throws Exception {
    SparseVector vec1 = new SparseVector(3);
    SparseVector vec2 = new SparseVector(6);
    VectorView vecV1 = new VectorView(vec1, 0, 3);
    VectorView vecV2 = new VectorView(vec2, 2, 3);
    testVectors(vecV1, vecV2);
  }
}