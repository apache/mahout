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

package org.apache.mahout.vectorizer.encoders;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public final class ContinuousValueEncoderTest extends MahoutTestCase {
  
  @Test
  public void testAddToVector() {
    FeatureVectorEncoder enc = new ContinuousValueEncoder("foo");
    Vector v1 = new DenseVector(20);
    enc.addToVector("-123", v1);
    assertEquals(-123, v1.minValue(), 0);
    assertEquals(0, v1.maxValue(), 0);
    assertEquals(123, v1.norm(1), 0);

    v1 = new DenseVector(20);
    enc.addToVector("123", v1);
    assertEquals(123, v1.maxValue(), 0);
    assertEquals(0, v1.minValue(), 0);
    assertEquals(123, v1.norm(1), 0);

    Vector v2 = new DenseVector(20);
    enc.setProbes(2);
    enc.addToVector("123", v2);
    assertEquals(123, v2.maxValue(), 0);
    assertEquals(2 * 123, v2.norm(1), 0);

    // v1 has one probe, v2 has two.  The first probe in v2 should be in the same
    // place as the only probe in v1
    v1 = v2.minus(v1);
    assertEquals(123, v1.maxValue(), 0);
    assertEquals(123, v1.norm(1), 0);

    Vector v3 = new DenseVector(20);
    enc.setProbes(2);
    enc.addToVector("100", v3);
    v1 = v2.minus(v3);
    assertEquals(23, v1.maxValue(), 0);
    assertEquals(2 * 23, v1.norm(1), 0);

    enc.addToVector("7", v1);
    assertEquals(30, v1.maxValue(), 0);
    assertEquals(2 * 30, v1.norm(1), 0);
    assertEquals(30, v1.get(10), 0);
    assertEquals(30, v1.get(18), 0);

    v2 = new DenseVector(20);
    v3 = new DenseVector(20);
    enc.setProbes(6);
    enc.addToVector("145", v2);
    enc.addToVector((byte[]) null, 145, v3);
    assertEquals(0, v2.minus(v3).norm(1), 0);

    try {
      enc.addToVector("foobar", v1);
      fail("Should have noticed bad numeric format");
    } catch (NumberFormatException e) {
      assertEquals("For input string: \"foobar\"", e.getMessage());
    }
  }

  @Test
  public void testAsString() {
    FeatureVectorEncoder enc = new ContinuousValueEncoder("foo");
    assertEquals("foo:123", enc.asString("123"));
  }

}
