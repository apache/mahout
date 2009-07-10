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

package org.apache.mahout.cf.taste.impl.common;

import java.io.Serializable;

/**
 * <p>Like {@link org.apache.mahout.cf.taste.impl.common.FullRunningAverage} but uses smaller values (short, float) to
 * conserve memory. This is used in {@link org.apache.mahout.cf.taste.impl.recommender.slopeone.SlopeOneRecommender}
 * since it may allocate tens of millions of such objects.</p>
 */
public class CompactRunningAverage implements RunningAverage, Serializable {

  private char count;
  private float average;

  public CompactRunningAverage() {
    count = (char) 0;
    average = Float.NaN;
  }

  @Override
  public void addDatum(double datum) {
    if ((int) count < 65535) { // = 65535 = 2^16 - 1
      if ((int) ++count == 1) {
        average = (float) datum;
      } else {
        average = (float)
            ((double) average * ((double) ((int) count - 1) / (double) count) + datum / (double) count);
      }
    }
  }

  @Override
  public void removeDatum(double datum) {
    if ((int) count == 0) {
      throw new IllegalStateException();
    }
    if ((int) --count == 0) {
      average = Float.NaN;
    } else {
      average = (float)
          ((double) average * ((double) ((int) count + 1) / (double) count) - datum / (double) count);
    }
  }

  @Override
  public void changeDatum(double delta) {
    if ((int) count == 0) {
      throw new IllegalStateException();
    }
    average += (float) (delta / (double) count);
  }

  @Override
  public int getCount() {
    return (int) count;
  }

  @Override
  public double getAverage() {
    return (double) average;
  }

  @Override
  public String toString() {
    return String.valueOf(average);
  }

}
