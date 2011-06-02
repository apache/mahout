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

import java.util.Collections;
import java.util.Comparator;

/**
 * this class will preserve the k minimum elements of all elements it has been offered
 */
public class MinK<T> extends FixedSizePriorityQueue<T> {

  public MinK(int k, Comparator<? super T> comparator) {
    super(k, comparator);
  }

  @Override
  protected Comparator<? super T> queueingComparator(Comparator<? super T> stdComparator) {
    return Collections.reverseOrder(stdComparator);
  }

  @Override
  protected Comparator<? super T> sortingComparator(Comparator<? super T> stdComparator) {
    return stdComparator;
  }

  public T greatestSmall() {
    return peek();
  }
}
