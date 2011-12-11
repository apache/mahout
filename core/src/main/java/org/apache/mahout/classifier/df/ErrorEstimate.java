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

package org.apache.mahout.classifier.df;

import com.google.common.base.Preconditions;

/**
 * Various methods to compute from the output of a random forest
 */
public final class ErrorEstimate {

  private ErrorEstimate() {
  }
  
  public static double errorRate(double[] labels, double[] predictions) {
    Preconditions.checkArgument(labels.length == predictions.length, "labels.length != predictions.length");
    double nberrors = 0; // number of instance that got bad predictions
    double datasize = 0; // number of classified instances

    for (int index = 0; index < labels.length; index++) {
      if (predictions[index] == -1) {
        continue; // instance not classified
      }

      if (predictions[index] != labels[index]) {
        nberrors++;
      }

      datasize++;
    }

    return nberrors / datasize;
  }

}
