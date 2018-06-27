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

package org.apache.mahout.utils.vectors;

import java.util.Iterator;
import java.util.Random;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.CountingIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.DoubleFunction;

public final class RandomVectorIterable implements Iterable<Vector> {

  public enum VectorType {DENSE, SPARSE}

  private final int numItems;
  private final VectorType type;
  
  public RandomVectorIterable() {
    this(100, VectorType.SPARSE);
  }
  
  public RandomVectorIterable(int numItems) {
    this(numItems, VectorType.SPARSE);
  }
  
  public RandomVectorIterable(int numItems, VectorType type) {
    this.numItems = numItems;
    this.type = type;
  }
  
  @Override
  public Iterator<Vector> iterator() {
    return Iterators.transform(
        new CountingIterator(numItems),
        new Function<Integer, Vector>() {
          private final Random random = RandomUtils.getRandom();
          @Override
          public Vector apply(Integer dummy) {
            Vector result =
                type == VectorType.SPARSE ? new RandomAccessSparseVector(numItems) : new DenseVector(numItems);
            result.assign(new DoubleFunction() {
              @Override
              public double apply(double ignored) {
                return random.nextDouble();
              }
            });
            return result;
          }
        });
  }

}
