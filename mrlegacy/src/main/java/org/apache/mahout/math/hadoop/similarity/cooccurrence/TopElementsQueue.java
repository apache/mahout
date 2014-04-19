/*
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

package org.apache.mahout.math.hadoop.similarity.cooccurrence;

import com.google.common.collect.Lists;
import org.apache.lucene.util.PriorityQueue;

import java.util.Collections;
import java.util.List;

public class TopElementsQueue extends PriorityQueue<MutableElement> {

  private final int maxSize;

  private static final int SENTINEL_INDEX = Integer.MIN_VALUE;

  public TopElementsQueue(int maxSize) {
    super(maxSize);
    this.maxSize = maxSize;
  }

  public List<MutableElement> getTopElements() {
    List<MutableElement> topElements = Lists.newArrayListWithCapacity(maxSize);
    while (size() > 0) {
      MutableElement top = pop();
      // filter out "sentinel" objects necessary for maintaining an efficient priority queue
      if (top.index() != SENTINEL_INDEX) {
        topElements.add(top);
      }
    }
    Collections.reverse(topElements);
    return topElements;
  }

  @Override
  protected MutableElement getSentinelObject() {
    return new MutableElement(SENTINEL_INDEX, Double.MIN_VALUE);
  }

  @Override
  protected boolean lessThan(MutableElement e1, MutableElement e2) {
    return e1.get() < e2.get();
  }
}
