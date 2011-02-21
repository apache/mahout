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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.apache.mahout.common.RandomUtils;

/**
 * Sample a fixed number of elements from an Iterator. The results will appear in the original order at some
 * cost in time and memory relative to a FixedSizeSampler.
 */
public class StableFixedSizeSamplingIterator<T> extends DelegatingIterator<T> {
  
  public StableFixedSizeSamplingIterator(int size, Iterator<T> source) {
    super(buildDelegate(size, source));
  }
  
  private static <T> Iterator<T> buildDelegate(int size, Iterator<T> source) {
    List<Entry<T>> buf = new ArrayList<Entry<T>>(size);
    int sofar = 0;
    while (source.hasNext()) {
      T v = source.next();
      sofar++;
      if (buf.size() < size) {
        buf.add(new Entry<T>(sofar, v));
      } else {
        Random generator = RandomUtils.getRandom();
        int position = generator.nextInt(sofar);
        if (position < buf.size()) {
          buf.get(position).value = v;
        }
      }
    }
    
    Collections.sort(buf);
    return new DelegateIterator<T>(buf);
  }
  
  private static final class Entry<T> implements Comparable<Entry<T>> {
    
    private final int originalIndex;
    private T value;
    
    private Entry(int originalIndex, T value) {
      this.originalIndex = originalIndex;
      this.value = value;
    }
    
    @Override
    public boolean equals(Object other) {
      return other instanceof Entry<?> && originalIndex == ((Entry<?>) other).originalIndex;
    }
    
    @Override
    public int hashCode() {
      return originalIndex;
    }
    
    @Override
    public int compareTo(Entry<T> other) {
      if (originalIndex < other.originalIndex) {
        return -1;
      } else if (originalIndex > other.originalIndex) {
        return 1;
      } else {
        return 0;
      }
    }
  }
  
  private static final class DelegateIterator<T> implements Iterator<T> {
    
    private final Iterator<Entry<T>> iterator;
    
    private DelegateIterator(List<Entry<T>> buf) {
      iterator = buf.iterator();
    }
    
    @Override
    public boolean hasNext() {
      return iterator.hasNext();
    }
    
    @Override
    public T next() {
      return iterator.next().value;
    }
    
    @Override
    public void remove() {
      throw new UnsupportedOperationException("Can't change sampler contents");
    }
  }
}
