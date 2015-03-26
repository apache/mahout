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

package org.apache.mahout.common.iterator;

import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.google.common.collect.ForwardingIterator;
import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;

/**
 * Sample a fixed number of elements from an Iterator. The results can appear in any order.
 */
public final class FixedSizeSamplingIterator<T> extends ForwardingIterator<T> {

  private final Iterator<T> delegate;
  
  public FixedSizeSamplingIterator(int size, Iterator<T> source) {
    List<T> buf = Lists.newArrayListWithCapacity(size);
    int sofar = 0;
    Random random = RandomUtils.getRandom();
    while (source.hasNext()) {
      T v = source.next();
      sofar++;
      if (buf.size() < size) {
        buf.add(v);
      } else {
        int position = random.nextInt(sofar);
        if (position < buf.size()) {
          buf.set(position, v);
        }
      }
    }
    delegate = buf.iterator();
  }

  @Override
  protected Iterator<T> delegate() {
    return delegate;
  }

}
