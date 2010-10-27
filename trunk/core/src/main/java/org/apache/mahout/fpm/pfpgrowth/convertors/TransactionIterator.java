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

package org.apache.mahout.fpm.pfpgrowth.convertors;

import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.mahout.common.Pair;

/**
 * Iterates over a Transaction and outputs the transaction integer id mapping and the support of the
 * transaction
 * 
 * @param <AP>
 */
public class TransactionIterator<AP> implements Iterator<Pair<int[],Long>> {
  private final Map<AP,Integer> attributeIdMapping;
  
  private final Iterator<Pair<List<AP>,Long>> iterator;
  
  private final int[] transactionBuffer;
  
  public TransactionIterator(Iterator<Pair<List<AP>,Long>> iterator, Map<AP,Integer> attributeIdMapping) {
    this.attributeIdMapping = attributeIdMapping;
    this.iterator = iterator;
    transactionBuffer = new int[attributeIdMapping.size()];
  }
  
  @Override
  public final boolean hasNext() {
    return iterator.hasNext();
  }
  
  @Override
  public final Pair<int[],Long> next() {
    Pair<List<AP>,Long> transaction = iterator.next();
    int index = 0;
    
    for (AP attribute : transaction.getFirst()) {
      if (attributeIdMapping.containsKey(attribute)) {
        transactionBuffer[index++] = attributeIdMapping.get(attribute);
      }
    }
    
    int[] transactionList = new int[index];
    System.arraycopy(transactionBuffer, 0, transactionList, 0, index);
    return new Pair<int[],Long>(transactionList, transaction.getSecond());
    
  }
  
  @Override
  public final void remove() {
    iterator.remove();
  }
  
}
