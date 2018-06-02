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

public class CachingEncoderTest extends MahoutTestCase {

  private static final int CARDINALITY = 10;
  private static final String NAME = "name";
  private static final String WORD = "word";
  private static final String CONTINUOUSVAL = "123";

  @Test
  public void testCacheAreUsedStaticWord() {
    CachingStaticWordValueEncoder encoder = new CachingStaticWordValueEncoder(NAME, CARDINALITY);
    Vector v = new DenseVector(CARDINALITY);
    encoder.addToVector(WORD, v);
    assertFalse("testCacheAreUsedStaticWord: cache should have values", encoder.getCaches()[0].isEmpty());
  }

  @Test
  public void testCacheAreUsedContinuous() {
    CachingContinuousValueEncoder encoder = new CachingContinuousValueEncoder(NAME, CARDINALITY);
    Vector v = new DenseVector(CARDINALITY);
    encoder.addToVector(CONTINUOUSVAL, 1.0, v);
    assertFalse("testCacheAreUsedContinuous: cache should have values", encoder.getCaches()[0].isEmpty());
  }

}
