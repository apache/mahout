/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sgd;

import org.apache.mahout.examples.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.junit.Test;

public class ModelDissectorTest extends MahoutTestCase {
  @Test
  public void testCategoryOrdering() {
    ModelDissector.Weight w = new ModelDissector.Weight("a", new DenseVector(new double[]{-2, -5, 5, 2, 4, 1, 0}), 4);
    assertEquals(1, w.getCategory(0), 0);
    assertEquals(-5, w.getWeight(0), 0);

    assertEquals(2, w.getCategory(1), 0);
    assertEquals(5, w.getWeight(1), 0);

    assertEquals(4, w.getCategory(2), 0);
    assertEquals(4, w.getWeight(2), 0);

    assertEquals(0, w.getCategory(3), 0);
    assertEquals(-2, w.getWeight(3), 0);
  }
}
