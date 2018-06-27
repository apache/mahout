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

package org.apache.mahout.utils.vectors.lucene;

import org.apache.mahout.common.RandomUtils;

class TermInfoClusterInOut implements Comparable<TermInfoClusterInOut> {

  private final String term;
  private final int inClusterDF;
  private final int outClusterDF;
  private final double logLikelihoodRatio;

  TermInfoClusterInOut(String term, int inClusterDF, int outClusterDF, double logLikelihoodRatio) {
    this.term = term;
    this.inClusterDF = inClusterDF;
    this.outClusterDF = outClusterDF;
    this.logLikelihoodRatio = logLikelihoodRatio;
  }

  @Override
  public int hashCode() {
    return term.hashCode() ^ inClusterDF ^ outClusterDF ^ RandomUtils.hashDouble(logLikelihoodRatio);
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof TermInfoClusterInOut)) {
      return false;
    }
    TermInfoClusterInOut other = (TermInfoClusterInOut) o;
    return term.equals(other.getTerm())
        && inClusterDF == other.getInClusterDF()
        && outClusterDF == other.getOutClusterDF()
        && logLikelihoodRatio == other.getLogLikelihoodRatio();
  }

  @Override
  public int compareTo(TermInfoClusterInOut that) {
    int res = Double.compare(that.logLikelihoodRatio, logLikelihoodRatio);
    if (res == 0) {
      res = term.compareTo(that.term);
    }
    return res;
  }

  public int getInClusterDiff() {
    return this.inClusterDF - this.outClusterDF;
  }

  String getTerm() {
    return term;
  }

  int getInClusterDF() {
    return inClusterDF;
  }

  int getOutClusterDF() {
    return outClusterDF;
  }

  double getLogLikelihoodRatio() {
    return logLikelihoodRatio;
  }
}
