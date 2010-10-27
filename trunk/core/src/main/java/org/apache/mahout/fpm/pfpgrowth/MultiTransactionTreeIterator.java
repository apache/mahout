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

package org.apache.mahout.fpm.pfpgrowth;

import java.util.Iterator;
import java.util.List;

import org.apache.mahout.common.Pair;

/**
 * Iterates over multiple transaction trees to produce a single iterator of transactions
 * 
 */
public final class MultiTransactionTreeIterator implements Iterator<List<Integer>> {
  
  private Iterator<Pair<List<Integer>,Long>> pIterator;
  
  private Pair<List<Integer>,Long> currentPattern;
  
  private long currentCount;
  
  public MultiTransactionTreeIterator(Iterator<Pair<List<Integer>,Long>> iterator) {
    this.pIterator = iterator;
    
    if (pIterator.hasNext()) {
      currentPattern = pIterator.next();
      currentCount = 0;
    } else {
      pIterator = null;
    }
    
  }
  
  @Override
  public boolean hasNext() {
    return pIterator != null;
  }
  
  @Override
  public List<Integer> next() {
    List<Integer> returnable = currentPattern.getFirst();
    currentCount++;
    if (currentCount == currentPattern.getSecond()) {
      if (pIterator.hasNext()) {
        currentPattern = pIterator.next();
        currentCount = 0;
      } else {
        pIterator = null;
      }
    }
    return returnable;
  }
  
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }
  
}
