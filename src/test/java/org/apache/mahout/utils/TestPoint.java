package org.apache.mahout.utils;

import java.util.Arrays;

import junit.framework.TestCase;

/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

public class TestPoint extends TestCase {
  public void testFormatPoint() {
    assertEquals("[1.0, 1.5]", Point.formatPoint(new Float [] {1.0f, 1.5f}));
  }
  
  public void testPtOut() {
    assertEquals("abc[1.0, 1.5]", Point.ptOut("abc", new Float [] {1.0f, 1.5f}));
  }
  
  public void testDecodePoint() {
    assertEquals(
        Arrays.asList(new Float [] {1.0f, 2.5f}), 
        Arrays.asList(Point.decodePoint("[1.0, 2.5]")));
  }
  
  public void testDecodePointWithPayload() {
    assertEquals(
        Arrays.asList(new Float [] {1.0f, 2.5f}), 
        Arrays.asList(Point.decodePoint("[1.0, 2.5] payloadhere, blah [][]")));
  }
}
