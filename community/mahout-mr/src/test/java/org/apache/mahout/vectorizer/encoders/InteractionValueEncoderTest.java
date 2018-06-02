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

import java.util.Map;
import java.util.Set;

import static com.google.common.collect.Iterables.getFirst;

import com.google.common.collect.Maps;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class InteractionValueEncoderTest extends MahoutTestCase {
  @Test
  public void testAddToVector() {
    WordValueEncoder wv = new StaticWordValueEncoder("word");
    ContinuousValueEncoder cv = new ContinuousValueEncoder("cont");
    InteractionValueEncoder enc = new InteractionValueEncoder("interactions", wv, cv);
    Vector v1 = new DenseVector(200);
    enc.addInteractionToVector("a","1.0",1.0, v1);
    int k = enc.getProbes();
    // should set k distinct locations to 1
    assertEquals((float) k, v1.norm(1), 0);
    assertEquals(1.0, v1.maxValue(), 0);

    // adding same interaction again should increment weights
    enc.addInteractionToVector("a","1.0",1.0,v1);
    assertEquals((float) k*2, v1.norm(1), 0);
    assertEquals(2.0, v1.maxValue(), 0);

    Vector v2 = new DenseVector(20000);
    enc.addInteractionToVector("a","1.0",1.0,v2);
    wv.addToVector("a", v2);
    cv.addToVector("1.0", v2);
    k = enc.getProbes();
    //this assumes no hash collision
    assertEquals((float) (k + wv.getProbes()+cv.getProbes()), v2.norm(1), 1.0e-3);
  }

  @Test
  public void testAddToVectorUsesProductOfWeights() {
    WordValueEncoder wv = new StaticWordValueEncoder("word");
    ContinuousValueEncoder cv = new ContinuousValueEncoder("cont");
    InteractionValueEncoder enc = new InteractionValueEncoder("interactions", wv, cv);
    Vector v1 = new DenseVector(200);
    enc.addInteractionToVector("a","0.9",0.5, v1);
    int k = enc.getProbes();
    // should set k distinct locations to 0.9*0.5
    assertEquals((float) k*0.5*0.9, v1.norm(1), 0);
    assertEquals(0.5*0.9, v1.maxValue(), 0);
  }

  @Test
  public void testAddToVectorWithTextValueEncoder() {
    WordValueEncoder wv = new StaticWordValueEncoder("word");
    TextValueEncoder tv = new TextValueEncoder("text");
    InteractionValueEncoder enc = new InteractionValueEncoder("interactions", wv, tv);
    Vector v1 = new DenseVector(200);
    enc.addInteractionToVector("a","some text here",1.0, v1);
    int k = enc.getProbes();
    // should interact "a" with each of "some","text" and "here"
    assertEquals((float) k*3, v1.norm(1), 0);
  }
  
  @Test
  public void testTraceDictionary() {
    StaticWordValueEncoder encoder1 = new StaticWordValueEncoder("first");
    StaticWordValueEncoder encoder2 = new StaticWordValueEncoder("second");
    
    Map<String, Set<Integer>> traceDictionary = Maps.newHashMap();

    InteractionValueEncoder interactions = new InteractionValueEncoder("interactions", encoder1, encoder2);
    interactions.setProbes(1);
    interactions.setTraceDictionary(traceDictionary);
    
    Vector v = new DenseVector(10);
    interactions.addInteractionToVector("a", "b", 1, v);
    
    assertEquals(1, v.getNumNonZeroElements());
    assertEquals(1, traceDictionary.size());
    assertEquals("interactions=a:b", getFirst(traceDictionary.keySet(), null));

  }

}
