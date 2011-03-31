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
import org.apache.mahout.common.iterator.TransformingIterator;

/**
 * Iterates over a Transaction and outputs the transaction integer id mapping and the support of the
 * transaction
 */
public class TransactionIterator<T> extends TransformingIterator<Pair<List<T>,Long>,Pair<int[],Long>> {

  private final Map<T,Integer> attributeIdMapping;
  private final int[] transactionBuffer;
  
  public TransactionIterator(Iterator<Pair<List<T>,Long>> iterator, Map<T,Integer> attributeIdMapping) {
    super(iterator);
    this.attributeIdMapping = attributeIdMapping;
    transactionBuffer = new int[attributeIdMapping.size()];
  }

  @Override
  protected Pair<int[],Long> transform(Pair<List<T>, Long> in) {
    int index = 0;
    for (T attribute : in.getFirst()) {
      if (attributeIdMapping.containsKey(attribute)) {
        transactionBuffer[index++] = attributeIdMapping.get(attribute);
      }
    }
    int[] transactionList = new int[index];
    System.arraycopy(transactionBuffer, 0, transactionList, 0, index);
    return new Pair<int[],Long>(transactionList, in.getSecond());
  }

  
}
