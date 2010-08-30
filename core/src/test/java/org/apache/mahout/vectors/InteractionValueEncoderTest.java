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

import com.google.common.collect.ImmutableMap;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Assert;
import org.junit.Test;

import java.util.Locale;

import static org.junit.Assert.assertEquals;

public class InteractionValueEncoderTest {
  @Test
  public void testAddToVector() {
    InteractionValueEncoder enc = new InteractionValueEncoder("interactions");
    Vector v1 = new DenseVector(200);
    enc.addInteractionToVector("a","b",v1);
    int k = enc.getProbes();
    // should set k distinct locations to 1
    Assert.assertEquals((float) k, v1.norm(1), 0);
    Assert.assertEquals(1.0, v1.maxValue(), 0);
    // adding same interaction again should increment weights
    enc.addInteractionToVector("a","b",v1);
    Assert.assertEquals((float) k*2, v1.norm(1), 0);
    Assert.assertEquals(2.0, v1.maxValue(), 0);

    Vector v2 = new DenseVector(20000);
    StaticWordValueEncoder wordEncoder = new StaticWordValueEncoder("test");
    enc.addInteractionToVector("a","b",v2);
    wordEncoder.addToVector("a", v2);
    wordEncoder.addToVector("b", v2);
    k = enc.getProbes();
    int j = wordEncoder.getProbes();
    //this assumes no hash collision
    Assert.assertEquals((float) (k + 2*j), v2.norm(1), 0);
  }

}
