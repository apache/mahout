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

import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

/**
 * {@link FrequentPatternMaxHeap} keeps top K Attributes in a TreeSet
 * 
 */
public class FrequentPatternMaxHeap {
  private final Comparator<Pattern> treeSetComparator = new Comparator<Pattern>() {
    @Override
    public int compare(Pattern cr1, Pattern cr2) {
      long support2 = cr2.support();
      long support1 = cr1.support();
      int length2 = cr2.length();
      int length1 = cr1.length();
      if (support1 == support2) {
        if (length1 == length2) {// if they are of same length and support order
          // randomly
          return 1;
        } else
          return length2 - length1;
      } else {
        if (support2 - support1 > 0)
          return 1;
        else
          return -1;
      }
    }
  };

  private int count = 0;

  private Pattern least = null;

  private int maxSize = 0;

  private HashMap<Long, Set<Pattern>> patternIndex = null;

  private TreeSet<Pattern> set = null;

  public FrequentPatternMaxHeap(int numResults) {
    maxSize = numResults;
    set = new TreeSet<Pattern>(treeSetComparator);
  }

  public final boolean addable(long support) {
    if (count < maxSize)
      return true;
    return least.support() <= support;
  }

  public final TreeSet<Pattern> getHeap() {
    return set;
  }

  public final void insert(Pattern frequentPattern) {
    insert(frequentPattern, true);
  }

  public final void insert(Pattern frequentPattern, boolean subPatternCheck) {
    if (subPatternCheck)// lazy initialization
    {
      if (patternIndex == null)
        patternIndex = new HashMap<Long, Set<Pattern>>();
    }
    if (count == maxSize) {
      int cmp = treeSetComparator.compare(frequentPattern, least);
      if (cmp < 0) {
        if (addPattern(frequentPattern, subPatternCheck)) {
          Pattern evictedItem = set.pollLast();
          least = set.last();
          if (subPatternCheck)
            patternIndex.get(evictedItem.support()).remove(evictedItem);
        }
      }
    } else {
      if (addPattern(frequentPattern, subPatternCheck)) {
        count++;
        if (least != null) {
          int cmp = treeSetComparator.compare(least, frequentPattern);
          if (cmp < 0)
            least = frequentPattern;
        } else if (least == null)
          least = frequentPattern;
      }
    }
  }
  
  public final int count(){
    return count;
  }

  public final boolean isFull() {
    return count == maxSize;
  }

  public final long leastSupport() {
    if (least == null)
      return 0;
    return least.support();
  }

  @Override
  public final String toString() {
    return super.toString();
  }

  private boolean addPattern(Pattern frequentPattern,
      boolean subPatternCheck) {
    if (subPatternCheck == false) {
      set.add(frequentPattern);
      return true;
    } else {
      Long index = frequentPattern.support();
      if (patternIndex.containsKey(index)) {
        Set<Pattern> indexSet = patternIndex.get(index);
        boolean replace = false;
        Pattern replacablePattern = null;
        for (Pattern p : indexSet) {

          if (frequentPattern.isSubPatternOf(p))
            return false;
          else if (p.isSubPatternOf(frequentPattern)) {
            replace = true;
            replacablePattern = p;
            break;
          }
        }
        if (replace) {
          indexSet.remove(replacablePattern);
          if (set.remove(replacablePattern))
            count--;
          if (indexSet.contains(frequentPattern) == false) {
            if (set.add(frequentPattern))
              count++;
            indexSet.add(frequentPattern);
          }
          return false;
        }
        set.add(frequentPattern);
        indexSet.add(frequentPattern);
        return true;
      } else {
        set.add(frequentPattern);
        Set<Pattern> patternList;
        if (patternIndex.containsKey(index) == false) {
          patternList = new HashSet<Pattern>();
          patternIndex.put(index, patternList);
        }
        patternList = patternIndex.get(index);
        patternList.add(frequentPattern);

        return true;
      }
    }
  }
}
