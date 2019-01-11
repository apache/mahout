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

package org.apache.mahout.math.neighborhood;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.WeightedVector;

/**
 * Decorates a weighted vector with a locality sensitive hash.
 *
 * The LSH function implemented is the random hyperplane based hash function.
 * See "Similarity Estimation Techniques from Rounding Algorithms" by Moses S. Charikar, section 3.
 * http://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/CharikarEstim.pdf
 */
public class HashedVector extends WeightedVector {
  protected static final int INVALID_INDEX = -1;

  /**
   * Value of the locality sensitive hash. It is 64 bit.
   */
  private final long hash;

  public HashedVector(Vector vector, long hash, int index) {
    super(vector, 1, index);
    this.hash = hash;
  }

  public HashedVector(Vector vector, Matrix projection, int index, long mask) {
    super(vector, 1, index);
    this.hash = mask & computeHash64(vector, projection);
  }

  public HashedVector(WeightedVector weightedVector, Matrix projection, long mask) {
    super(weightedVector.getVector(), weightedVector.getWeight(), weightedVector.getIndex());
    this.hash = mask & computeHash64(weightedVector, projection);
  }

  public static long computeHash64(Vector vector, Matrix projection) {
    long hash = 0;
    for (Element element : projection.times(vector).nonZeroes()) {
      if (element.get() > 0) {
        hash += 1L << element.index();
      }
    }
    return hash;
  }

  public static HashedVector hash(WeightedVector v, Matrix projection) {
    return hash(v, projection, 0);
  }

  public static HashedVector hash(WeightedVector v, Matrix projection, long mask) {
    return new HashedVector(v, projection, mask);
  }

  public int hammingDistance(long otherHash) {
    return Long.bitCount(hash ^ otherHash);
  }

  public long getHash() {
    return hash;
  }

  @Override
  public String toString() {
    return String.format("index=%d, hash=%08x, v=%s", getIndex(), hash, getVector());
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof HashedVector)) {
      return o instanceof Vector && this.minus((Vector) o).norm(1) == 0;
    }
    HashedVector v = (HashedVector) o;
    return v.hash == this.hash && this.minus(v).norm(1) == 0;
  }

  @Override
  public int hashCode() {
    int result = super.hashCode();
    result = 31 * result + (int) (hash ^ (hash >>> 32));
    return result;
  }
}
