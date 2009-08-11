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

package org.apache.mahout.cf.taste.impl.transforms;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.transforms.SimilarityTransform;

import java.util.Collection;

/**
 * <p>Applies "case amplification" to similarities. This essentially makes big values bigger and small values smaller by
 * raising each score to a power. It could however be used to achieve the opposite effect.</p>
 */
public final class CaseAmplification implements SimilarityTransform {

  private final double factor;

  /**
   * <p>Creates a {@link CaseAmplification} transformation based on the given factor.</p>
   *
   * @param factor transformation factor
   * @throws IllegalArgumentException if factor is 0.0 or {@link Double#NaN}
   */
  public CaseAmplification(double factor) {
    if (Double.isNaN(factor) || factor == 0.0) {
      throw new IllegalArgumentException("factor is 0 or NaN");
    }
    this.factor = factor;
  }

  /**
   * <p>Transforms one similarity value. This implementation is such that it's possible to define this transformation on
   * one value in isolation. The "thing" parameters are therefore unused.</p>
   *
   * @param id1 unused
   * @param id2 unused
   * @param value  similarity to transform
   * @return <code>value<sup>factor</sup></code> if value is nonnegative; <code>-value<sup>-factor</sup></code>
   *         otherwise
   */
  @Override
  public double transformSimilarity(long id1, long id2, double value) {
    return value < 0.0 ? -Math.pow(-value, factor) : Math.pow(value, factor);
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

  @Override
  public String toString() {
    return "CaseAmplification[factor:" + factor + ']';
  }

}
