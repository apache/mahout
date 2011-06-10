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

package org.apache.mahout.cf.taste.common;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

/**
 * base class for queues holding the top or min k elements of all elements they have been offered
 */
abstract class FixedSizePriorityQueue<T> {

  private final int k;
  private final Comparator<? super T> comparator;
  private final Queue<T> queue;

  FixedSizePriorityQueue(int k, Comparator<? super T> comparator) {
    Preconditions.checkArgument(k > 0);
    this.k = k;
    this.comparator = Preconditions.checkNotNull(comparator);
    this.queue = new PriorityQueue<T>(k + 1, queueingComparator(comparator));
  }

  abstract Comparator<? super T> queueingComparator(Comparator<? super T> stdComparator);
  abstract Comparator<? super T> sortingComparator(Comparator<? super T> stdComparator);

  public void offer(T item) {
    if (queue.size() < k) {
      queue.add(item);
    } else if (comparator.compare(item, queue.peek()) > 0) {
      queue.add(item);
      queue.poll();
    }
  }

  public boolean isEmpty() {
    return queue.isEmpty();
  }

  public int size() {
    return queue.size();
  }

  public List<T> retrieve() {
    List<T> topItems = Lists.newArrayList(queue);
    Collections.sort(topItems, sortingComparator(comparator));
    return topItems;
  }

  protected T peek() {
    return queue.peek();
  }
}
