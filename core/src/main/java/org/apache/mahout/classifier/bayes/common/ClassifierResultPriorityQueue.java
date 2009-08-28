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

package org.apache.mahout.classifier.bayes.common;

import org.apache.hadoop.util.PriorityQueue;
import org.apache.mahout.classifier.ClassifierResult;

public class ClassifierResultPriorityQueue extends PriorityQueue<ClassifierResult> {

  public ClassifierResultPriorityQueue(int numResults) {
    initialize(numResults);
  }

  @Override
  protected boolean lessThan(Object a, Object b) {
    ClassifierResult cr1 = (ClassifierResult) a;
    ClassifierResult cr2 = (ClassifierResult) b;

    double score1 = cr1.getScore();
    double score2 = cr2.getScore();
    return score1 == score2 ? cr1.getLabel().compareTo(cr2.getLabel()) < 0 : score2<score1;
  }
}
