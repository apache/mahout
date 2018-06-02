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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.impl.TasteTestCase;

abstract class SimilarityTestCase extends TasteTestCase {

  static void assertCorrelationEquals(double expected, double actual) {
    if (Double.isNaN(expected)) {
      assertTrue("Correlation is not NaN", Double.isNaN(actual));
    } else {
      assertTrue("Correlation is NaN", !Double.isNaN(actual));
      assertTrue("Correlation > 1.0", actual <= 1.0);
      assertTrue("Correlation < -1.0", actual >= -1.0);
      assertEquals(expected, actual, EPSILON);
    }
  }

}
