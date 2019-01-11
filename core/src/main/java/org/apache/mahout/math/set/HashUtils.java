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

package org.apache.mahout.math.set;

/**
 * Computes hashes of primitive values.  Providing these as statics allows the templated code
 * to compute hashes of sets.
 */
public final class HashUtils {

  private HashUtils() {
  }

  public static int hash(byte x) {
    return x;
  }

  public static int hash(short x) {
    return x;
  }

  public static int hash(char x) {
    return x;
  }

  public static int hash(int x) {
    return x;
  }

  public static int hash(float x) {
    return Float.floatToIntBits(x) >>> 3 + Float.floatToIntBits((float) (Math.PI * x));
  }

  public static int hash(double x) {
    return hash(17 * Double.doubleToLongBits(x));
  }

  public static int hash(long x) {
    return (int) ((x * 11) >>> 32 ^ x);
  }
}
