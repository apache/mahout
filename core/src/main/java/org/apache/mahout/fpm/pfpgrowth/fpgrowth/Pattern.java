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

package org.apache.mahout.fpm.pfpgrowth.fpgrowth;

import java.util.Arrays;

public class Pattern {

  private static final int DEFAULT_INITIAL_SIZE = 2;

  private static final float GROWTH_RATE = 1.5f;

  private boolean dirty = true;

  private int hashCode;

  private int length = 0;

  private int[] pattern;

  private long support = Long.MAX_VALUE;

  private long[] supportValues;

  public Pattern() {
    this(DEFAULT_INITIAL_SIZE);
  }

  private Pattern(int size) {
    if (size < DEFAULT_INITIAL_SIZE)
      size = DEFAULT_INITIAL_SIZE;
    this.pattern = new int[size];
    this.supportValues = new long[size];
    dirty = true;
  }

  public final void add(int id, long support) {
    if (length >= pattern.length)
      resize();
    this.pattern[length] = id;
    this.supportValues[length++] = support;
    this.support = (support > this.support) ? this.support : support;
    dirty = true;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    Pattern other = (Pattern) obj;
    if (length != other.length)
      return false;
    if (support != other.support)
      return false;
    if (!Arrays.equals(pattern, other.pattern))
      return false;
    return true;
  }

  public final int[] getPattern() {
    return this.pattern;
  }

  public final Object[] getPatternWithSupport() {
    return new Object[] { this.pattern, this.supportValues };
  }

  @Override
  public int hashCode() {
    if (dirty == false)
      return hashCode;
    int result = 1;
    int prime = 31;
    result = prime * result + Arrays.hashCode(pattern);
    result = prime * result + Long.valueOf(support).hashCode();
    hashCode = result;
    return result;
  }

  public final boolean isSubPatternOf(Pattern frequentPattern) {
    int[] otherPattern = frequentPattern.getPattern();
    int otherLength = frequentPattern.length();
    if (this.length() > frequentPattern.length())
      return false;
    int i = 0;
    int otherI = 0;
    while (i < length && otherI < otherLength) {
      if (otherPattern[otherI] == pattern[i]) {
        otherI++;
        i++;
      } else if (otherPattern[otherI] < pattern[i]) {
        otherI++;
      } else
        return false;
    }
    return otherI != otherLength || i == length;
  }

  public final int length() {
    return this.length;
  }

  public final long support() {
    return this.support;
  }

  @Override
  public final String toString() {
    int[] arr = new int[length];
    System.arraycopy(pattern, 0, arr, 0, length);
    return Arrays.toString(arr) + '-' + support;
  }

  private void resize() {
    int size = (int) (GROWTH_RATE * length);
    if (size < DEFAULT_INITIAL_SIZE)
      size = DEFAULT_INITIAL_SIZE;
    int[] oldpattern = pattern;
    long[] oldSupport = supportValues;
    this.pattern = new int[size];
    this.supportValues = new long[size];
    System.arraycopy(oldpattern, 0, this.pattern, 0, length);
    System.arraycopy(oldSupport, 0, this.supportValues, 0, length);
  }

}
