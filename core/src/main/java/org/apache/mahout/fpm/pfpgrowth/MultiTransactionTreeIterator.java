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

import com.google.common.collect.AbstractIterator;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.list.IntArrayList;

/**
 * Iterates over multiple transaction trees to produce a single iterator of transactions
 */
public final class MultiTransactionTreeIterator extends AbstractIterator<IntArrayList> {
  
  private final Iterator<Pair<IntArrayList,Long>> pIterator;
  private IntArrayList current;
  private long currentMaxCount;
  private long currentCount;
  
  public MultiTransactionTreeIterator(Iterator<Pair<IntArrayList,Long>> iterator) {
    this.pIterator = iterator;
  }

  @Override
  protected IntArrayList computeNext() {
    if (currentCount >= currentMaxCount) {
      if (pIterator.hasNext()) {
        Pair<IntArrayList,Long> nextValue = pIterator.next();
        current = nextValue.getFirst();
        currentMaxCount = nextValue.getSecond();
        currentCount = 0;
      } else {
        return endOfData();
      }
    }
    currentCount++;
    return current;
  }
  
}
