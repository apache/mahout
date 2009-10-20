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

import org.apache.mahout.matrix.Vector;
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.UnaryFunction;
import org.apache.mahout.matrix.SparseVector;
import org.apache.mahout.common.RandomUtils;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

public class RandomVectorIterable implements Iterable<Vector>{

  private int numItems = 100;
  public enum VectorType {DENSE, SPARSE}

  private VectorType type = VectorType.SPARSE;

  public RandomVectorIterable() {
  }

  public RandomVectorIterable(int numItems) {
    this.numItems = numItems;
  }

  public RandomVectorIterable(int numItems, VectorType type) {
    this.numItems = numItems;
    this.type = type;
  }

  @Override
  public Iterator<Vector> iterator() {
    return new VectIterator();
  }

  private class VectIterator implements Iterator<Vector>{
    private int count = 0;
    private final Random random = RandomUtils.getRandom();
    @Override
    public boolean hasNext() {
      return count < numItems;
    }

    @Override
    public Vector next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Vector result = type == VectorType.SPARSE ? new SparseVector(numItems) : new DenseVector(numItems);
      result.assign(new UnaryFunction(){
        @Override
        public double apply(double arg1) {
          return random.nextDouble();
        }
      });
      count++;
      return result;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
}
