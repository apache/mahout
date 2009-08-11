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

package org.apache.mahout.cf.taste.impl.transforms;

/** <p>Tests {@link CaseAmplification}.</p> */
public final class CaseAmplificationTest extends TransformTestCase {

  public void testOneValue() {
    assertEquals(2.0, new CaseAmplification(0.5).transformSimilarity(0, 0, 4.0), EPSILON);
    assertEquals(-2.0, new CaseAmplification(0.5).transformSimilarity(0, 0, -4.0), EPSILON);
  }

  public void testRefresh() {
    // Make sure this doesn't throw an exception
    new CaseAmplification(1.0).refresh(null);
  }

}
