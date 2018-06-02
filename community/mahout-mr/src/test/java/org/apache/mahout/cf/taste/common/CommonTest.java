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

package org.apache.mahout.cf.taste.common;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.PrintWriter;

/** <p>Tests common classes.</p> */
public final class CommonTest extends TasteTestCase {

  @Test
  public void testTasteException() {
    // Just make sure this all doesn't, ah, throw an exception
    TasteException te1 = new TasteException();
    TasteException te2 = new TasteException(te1);
    TasteException te3 = new TasteException(te2.toString(), te2);
    TasteException te4 = new TasteException(te3.toString());
    te4.printStackTrace(new PrintStream(new ByteArrayOutputStream()));
    te4.printStackTrace(new PrintWriter(new OutputStreamWriter(new ByteArrayOutputStream())));
  }

  @Test
  public void testNSUException() {
    // Just make sure this all doesn't, ah, throw an exception
    TasteException te1 = new NoSuchUserException();
    TasteException te4 = new NoSuchUserException(te1.toString());
    te4.printStackTrace(new PrintStream(new ByteArrayOutputStream()));
    te4.printStackTrace(new PrintWriter(new OutputStreamWriter(new ByteArrayOutputStream())));
  }

  @Test
  public void testNSIException() {
    // Just make sure this all doesn't, ah, throw an exception
    TasteException te1 = new NoSuchItemException();
    TasteException te4 = new NoSuchItemException(te1.toString());
    te4.printStackTrace(new PrintStream(new ByteArrayOutputStream()));
    te4.printStackTrace(new PrintWriter(new OutputStreamWriter(new ByteArrayOutputStream())));
  }

}
