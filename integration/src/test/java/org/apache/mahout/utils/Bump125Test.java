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

package org.apache.mahout.utils;

import com.google.common.collect.Lists;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

import java.util.Iterator;

public class Bump125Test extends MahoutTestCase {
  @Test
  public void testIncrement() throws Exception {
    Iterator<Integer> ref = Lists.newArrayList(1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60,
            70, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350,
            400, 500, 600, 700, 800, 1000, 1200, 1400, 1600, 1800,
            2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000)
            .iterator();
    Bump125 b = new Bump125();
    for (int i = 0; i < 50; i++) {
      long x = b.increment();
      assertEquals(ref.next().longValue(), x);
    }
  }
}
