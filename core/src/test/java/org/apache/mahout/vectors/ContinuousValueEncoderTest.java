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

package org.apache.mahout.vectors;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Assert;
import org.junit.Test;

public class ContinuousValueEncoderTest {
  @Test
  public void testAddToVector() {
    FeatureVectorEncoder enc = new ContinuousValueEncoder("foo");
    Vector v1 = new DenseVector(20);
    enc.addToVector("-123", v1);
    Assert.assertEquals(-123, v1.minValue(), 0);
    Assert.assertEquals(0, v1.maxValue(), 0);
    Assert.assertEquals(123, v1.norm(1), 0);

    v1 = new DenseVector(20);
    enc.addToVector("123", v1);
    Assert.assertEquals(123, v1.maxValue(), 0);
    Assert.assertEquals(0, v1.minValue(), 0);
    Assert.assertEquals(123, v1.norm(1), 0);

    Vector v2 = new DenseVector(20);
    enc.setProbes(2);
    enc.addToVector("123", v2);
    Assert.assertEquals(123, v2.maxValue(), 0);
    Assert.assertEquals(2 * 123, v2.norm(1), 0);

    v1 = v2.minus(v1);
    Assert.assertEquals(123, v1.maxValue(), 0);
    Assert.assertEquals(123, v1.norm(1), 0);

    Vector v3 = new DenseVector(20);
    enc.setProbes(2);
    enc.addToVector("100", v3);
    v1 = v2.minus(v3);
    Assert.assertEquals(23, v1.maxValue(), 0);
    Assert.assertEquals(2 * 23, v1.norm(1), 0);

    enc.addToVector("7", v1);
    Assert.assertEquals(30, v1.maxValue(), 0);
    Assert.assertEquals(2 * 30, v1.norm(1), 0);
    Assert.assertEquals(30, v1.get(10), 0);
    Assert.assertEquals(30, v1.get(18), 0);

    try {
      enc.addToVector("foobar", v1);
      Assert.fail("Should have noticed bad numeric format");
    } catch (NumberFormatException e) {
      Assert.assertEquals("For input string: \"foobar\"", e.getMessage());
    }
  }

  @Test
  public void testAsString() {
    ContinuousValueEncoder enc = new ContinuousValueEncoder("foo");
    Assert.assertEquals("foo:123", enc.asString("123"));
  }

}
