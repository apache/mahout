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

/**
 * Each entry in a {@link TermInfo} dictionary. Contains information about a term.
 */
public class TermEntry {

  private final String term;
  private final int termIdx;
  private final int docFreq;
  
  public TermEntry(String term, int termIdx, int docFreq) {
    this.term = term;
    this.termIdx = termIdx;
    this.docFreq = docFreq;
  }

  public String getTerm() {
    return term;
  }

  public int getTermIdx() {
    return termIdx;
  }

  public int getDocFreq() {
    return docFreq;
  }
}
