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

package org.apache.mahout.math.hadoop.similarity.cooccurrence.measures;

public class TanimotoCoefficientSimilarity extends CountbasedMeasure {

  @Override
  public double similarity(double dots, double normA, double normB, int numberOfColumns) {
    // Return 0 even when dots == 0 since this will cause it to be ignored -- not NaN
    return dots / (normA + normB - dots);
  }

  @Override
  public boolean consider(int numNonZeroEntriesA, int numNonZeroEntriesB, double maxValueA, double maxValueB,
      double threshold) {
    return numNonZeroEntriesA >= numNonZeroEntriesB * threshold
        && numNonZeroEntriesB >= numNonZeroEntriesA * threshold;
  }
}
