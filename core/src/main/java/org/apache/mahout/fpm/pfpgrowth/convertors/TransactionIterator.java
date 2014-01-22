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

import com.google.common.base.Function;
import com.google.common.collect.ForwardingIterator;
import com.google.common.collect.Iterators;
import org.apache.mahout.common.Pair;

/**
 * Iterates over a Transaction and outputs the transaction integer id mapping and the support of the
 * transaction
 */
public class TransactionIterator<T> extends ForwardingIterator<Pair<int[],Long>> {

  private final int[] transactionBuffer;
  private final Iterator<Pair<int[],Long>> delegate;

  public TransactionIterator(Iterator<Pair<List<T>,Long>> transactions, final Map<T,Integer> attributeIdMapping) {
    transactionBuffer = new int[attributeIdMapping.size()];
    delegate = Iterators.transform(
        transactions,
        new Function<Pair<List<T>,Long>, Pair<int[],Long>>() {
          @Override
          public Pair<int[],Long> apply(Pair<List<T>,Long> from) {
            if (from == null) {
              return null;
            }
            int index = 0;
            for (T attribute : from.getFirst()) {
              if (attributeIdMapping.containsKey(attribute)) {
                transactionBuffer[index++] = attributeIdMapping.get(attribute);
              }
            }
            int[] transactionList = new int[index];
            System.arraycopy(transactionBuffer, 0, transactionList, 0, index);
            return new Pair<int[],Long>(transactionList, from.getSecond());
          }
        });
  }

  @Override
  protected Iterator<Pair<int[],Long>> delegate() {
    return delegate;
  }

}
