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

import java.util.PriorityQueue;
import java.util.Set;

import com.google.common.collect.Sets;
import org.apache.mahout.math.map.OpenLongObjectHashMap;

/**  keeps top K Attributes in a TreeSet */
public final class FrequentPatternMaxHeap {
  
  private int count;
  private Pattern least;
  private final int maxSize;
  private final boolean subPatternCheck;
  private final OpenLongObjectHashMap<Set<Pattern>> patternIndex;
  private final PriorityQueue<Pattern> queue;
  
  public FrequentPatternMaxHeap(int numResults, boolean subPatternCheck) {
    maxSize = numResults;
    queue = new PriorityQueue<Pattern>(maxSize);
    this.subPatternCheck = subPatternCheck;
    patternIndex = new OpenLongObjectHashMap<Set<Pattern>>();
    for (Pattern p : queue) {
      Long index = p.support();
      Set<Pattern> patternList;
      if (!patternIndex.containsKey(index)) {
        patternList = Sets.newHashSet();
        patternIndex.put(index, patternList);
      }
      patternList = patternIndex.get(index);
      patternList.add(p);
      
    }
  }
  
  public boolean addable(long support) {
    return count < maxSize || least.support() <= support;
  }
  
  public PriorityQueue<Pattern> getHeap() {
    if (subPatternCheck) {
      PriorityQueue<Pattern> ret = new PriorityQueue<Pattern>(maxSize);
      for (Pattern p : queue) {
        if (patternIndex.get(p.support()).contains(p)) {
          ret.add(p);
        }
      }
      return ret;
    }
    return queue;
  }
  
  public void addAll(FrequentPatternMaxHeap patterns,
                     int attribute,
                     long attributeSupport) {
    for (Pattern pattern : patterns.getHeap()) {
      long support = Math.min(attributeSupport, pattern.support());
      if (this.addable(support)) {
        pattern.add(attribute, support);
        this.insert(pattern);
      }
    }
  }
  
  public void insert(Pattern frequentPattern) {
    if (frequentPattern.length() == 0) {
      return;
    }
    
    if (count == maxSize) {
      if (frequentPattern.compareTo(least) > 0 && addPattern(frequentPattern)) {
        Pattern evictedItem = queue.poll();
        least = queue.peek();
        if (subPatternCheck) {
          patternIndex.get(evictedItem.support()).remove(evictedItem);
        }
      }
    } else {
      if (addPattern(frequentPattern)) {
        count++;
        if (least == null) {
          least = frequentPattern;
        } else {
          if (least.compareTo(frequentPattern) < 0) {
            least = frequentPattern;
          }
        }
      }
    }
  }
  
  public int count() {
    return count;
  }
  
  public boolean isFull() {
    return count == maxSize;
  }
  
  public long leastSupport() {
    if (least == null) {
      return 0;
    }
    return least.support();
  }
  
  private boolean addPattern(Pattern frequentPattern) {
    if (subPatternCheck) {
      Long index = frequentPattern.support();
      if (patternIndex.containsKey(index)) {
        Set<Pattern> indexSet = patternIndex.get(index);
        boolean replace = false;
        Pattern replacablePattern = null;
        for (Pattern p : indexSet) {
          if (frequentPattern.isSubPatternOf(p)) {
            return false;
          } else if (p.isSubPatternOf(frequentPattern)) {
            replace = true;
            replacablePattern = p;
            break;
          }
        }
        if (replace) {
          indexSet.remove(replacablePattern);
          if (!indexSet.contains(frequentPattern) && queue.add(frequentPattern)) {
            indexSet.add(frequentPattern);
          }
          return false;
        }
        queue.add(frequentPattern);
        indexSet.add(frequentPattern);
      } else {
        queue.add(frequentPattern);
        Set<Pattern> patternList;
        if (!patternIndex.containsKey(index)) {
          patternList = Sets.newHashSet();
          patternIndex.put(index, patternList);
        }
        patternList = patternIndex.get(index);
        patternList.add(frequentPattern);
      }
    } else {
      queue.add(frequentPattern);
    }
    return true;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("FreqPatHeap{");
    String sep = "";
    for (Pattern p : getHeap()) {
      sb.append(sep).append(p);
      sep = ", ";
    }
    return sb.toString();
  }
}
