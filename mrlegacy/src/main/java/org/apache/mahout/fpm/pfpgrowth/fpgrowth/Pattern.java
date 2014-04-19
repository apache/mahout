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

import com.google.common.primitives.Longs;

import java.util.Arrays;

/**
 * A  in FPGrowth is a list of items (here int) and the
 * support(the number of times the pattern is seen in the dataset)
 * 
 */
public class Pattern implements Comparable<Pattern> {
  
  private static final int DEFAULT_INITIAL_SIZE = 2;
  
  private static final float GROWTH_RATE = 1.5f;
  
  private boolean dirty = true;
  
  private int hashCode;
  
  private int length;
  
  private int[] pattern;
  
  private long support = Long.MAX_VALUE;
  
  public Pattern() {
    this(DEFAULT_INITIAL_SIZE);
  }
  
  private Pattern(int size) {
    if (size < DEFAULT_INITIAL_SIZE) {
      size = DEFAULT_INITIAL_SIZE;
    }
    this.pattern = new int[size];
    dirty = true;
  }
  
  public final void add(int id, long supportCount) {
    dirty = true;
    if (length >= pattern.length) {
      resize();
    }
    this.pattern[length++] = id;
    Arrays.sort(this.pattern, 0, length);
    this.support = supportCount > this.support ? this.support : supportCount;
  }

  public final int[] getPattern() {
    return this.pattern;
  }

  public final boolean isSubPatternOf(Pattern frequentPattern) {
    int[] otherPattern = frequentPattern.getPattern();
    int otherLength = frequentPattern.length();
    if (this.length() > frequentPattern.length()) {
      return false;
    }
    int i = 0;
    int otherI = 0;
    while (i < length && otherI < otherLength) {
      if (otherPattern[otherI] == pattern[i]) {
        otherI++;
        i++;
      } else if (otherPattern[otherI] < pattern[i]) {
        otherI++;
      } else {
        return false;
      }
    }
    return otherI != otherLength || i == length;
  }

  public final int length() {
    return this.length;
  }

  public final long support() {
    return this.support;
  }

  private void resize() {
    int size = (int) (GROWTH_RATE * length);
    if (size < DEFAULT_INITIAL_SIZE) {
      size = DEFAULT_INITIAL_SIZE;
    }
    int[] oldpattern = pattern;
    this.pattern = new int[size];
    System.arraycopy(oldpattern, 0, this.pattern, 0, length);
  }
  
  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    Pattern other = (Pattern) obj;
    // expensive check done only if length and support matches    
    return length == other.length && support == other.support && Arrays.equals(pattern, other.pattern);
  }
  
  @Override
  public int hashCode() {
    if (!dirty) {
      return hashCode;
    }
    int result = Arrays.hashCode(pattern);
    result = 31 * result + Longs.hashCode(support);
    result = 31 * result + length;
    hashCode = result;
    return result;
  }
  
  @Override
  public final String toString() {
    int[] arr = new int[length];
    System.arraycopy(pattern, 0, arr, 0, length);
    return Arrays.toString(arr) + '-' + support;
  }
  
  @Override
  public int compareTo(Pattern cr2) {
    long support2 = cr2.support();
    int length2 = cr2.length();
    if (support == support2) {
      if (length < length2) {
        return -1;
      } else if (length > length2) {
        return 1;
      } else {
        return 0;
      }
    } else {
      return support > support2 ? 1 : -1;
    }
  }
  
}
