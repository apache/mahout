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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.util.Iterator;
import java.util.Locale;

public final class WordLikeValueEncoderTest extends MahoutTestCase {

  @Test
  public void testAddToVector() {
    FeatureVectorEncoder enc = new StaticWordValueEncoder("word");
    Vector v = new DenseVector(200);
    enc.addToVector("word1", v);
    enc.addToVector("word2", v);
    Iterator<Vector.Element> i = v.nonZeroes().iterator();
    Iterator<Integer> j = ImmutableList.of(7, 118, 119, 199).iterator();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      assertEquals(j.next().intValue(), element.index());
      assertEquals(1, element.get(), 0);
    }
    assertFalse(j.hasNext());
  }

  @Test
  public void testAsString() {
    Locale.setDefault(Locale.ENGLISH);
    FeatureVectorEncoder enc = new StaticWordValueEncoder("word");
    assertEquals("word:w1:1.0000", enc.asString("w1"));
  }

  @Test
  public void testStaticWeights() {
    StaticWordValueEncoder enc = new StaticWordValueEncoder("word");
    enc.setDictionary(ImmutableMap.<String, Double>of("word1", 3.0, "word2", 1.5));
    Vector v = new DenseVector(200);
    enc.addToVector("word1", v);
    enc.addToVector("word2", v);
    enc.addToVector("word3", v);
    Iterator<Vector.Element> i = v.nonZeroes().iterator();
    Iterator<Integer> j = ImmutableList.of(7, 101, 118, 119, 152, 199).iterator();
    Iterator<Double> k = ImmutableList.of(3.0, 0.75, 1.5, 1.5, 0.75, 3.0).iterator();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      assertEquals(j.next().intValue(), element.index());
    }
    i = v.nonZeroes().iterator();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      assertEquals(String.format("checking v[%d]", element.index()), k.next(), element.get(), 0);
    }
    assertFalse(j.hasNext());
  }

  @Test
  public void testDynamicWeights() {
    FeatureVectorEncoder enc = new AdaptiveWordValueEncoder("word");
    Vector v = new DenseVector(200);
    enc.addToVector("word1", v);  // weight is log(2/1.5)
    enc.addToVector("word2", v);  // weight is log(3.5 / 1.5)
    enc.addToVector("word1", v);  // weight is log(4.5 / 2.5) (but overlays on first value)
    enc.addToVector("word3", v);  // weight is log(6 / 1.5)
    Iterator<Vector.Element> i = v.nonZeroes().iterator();
    Iterator<Integer> j = ImmutableList.of(7, 101, 118, 119, 152, 199).iterator();
    Iterator<Double> k = ImmutableList.of(Math.log(2 / 1.5) + Math.log(4.5 / 2.5),
                                          Math.log(6 / 1.5), Math.log(3.5 / 1.5),
                                          Math.log(3.5 / 1.5), Math.log(6 / 1.5),
                                          Math.log(2 / 1.5) + Math.log(4.5 / 2.5)).iterator();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      assertEquals(j.next().intValue(), element.index());
      assertEquals(k.next(), element.get(), 1.0e-6);
    }
    assertFalse(j.hasNext());
  }
}
