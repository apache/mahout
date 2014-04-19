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

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import com.google.common.base.Function;
import com.google.common.collect.ForwardingIterator;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;

/**
 * Sample a fixed number of elements from an Iterator. The results will appear in the original order at some
 * cost in time and memory relative to a FixedSizeSampler.
 */
public class StableFixedSizeSamplingIterator<T> extends ForwardingIterator<T> {

  private final Iterator<T> delegate;
  
  public StableFixedSizeSamplingIterator(int size, Iterator<T> source) {
    List<Pair<Integer,T>> buf = Lists.newArrayListWithCapacity(size);
    int sofar = 0;
    Random random = RandomUtils.getRandom();
    while (source.hasNext()) {
      T v = source.next();
      sofar++;
      if (buf.size() < size) {
        buf.add(new Pair<Integer,T>(sofar, v));
      } else {
        int position = random.nextInt(sofar);
        if (position < buf.size()) {
          buf.set(position, new Pair<Integer,T>(sofar, v));
        }
      }
    }

    Collections.sort(buf);
    delegate = Iterators.transform(buf.iterator(),
      new Function<Pair<Integer,T>,T>() {
        @Override
        public T apply(Pair<Integer,T> from) {
          return from.getSecond();
        }
      });
  }

  @Override
  protected Iterator<T> delegate() {
    return delegate;
  }

}
