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

package org.apache.mahout.math.random;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;
import org.apache.mahout.common.RandomUtils;
import org.junit.Test;

import java.util.List;

public class IndianBuffetTest {
  @Test
  public void testBasicText() {
      RandomUtils.useTestSeed();
      IndianBuffet<String> sampler = IndianBuffet.createTextDocumentSampler(30);
      Multiset<String> counts = HashMultiset.create();
      int[] lengths = new int[100];
      for (int i = 0; i < 30; i++) {
          final List<String> doc = sampler.sample();
          lengths[doc.size()]++;
          for (String w : doc) {
              counts.add(w);
          }
          System.out.printf("%s\n", doc);
      }
  }
}
