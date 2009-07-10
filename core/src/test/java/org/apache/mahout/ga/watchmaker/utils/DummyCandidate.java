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

package org.apache.mahout.ga.watchmaker.utils;

import java.util.ArrayList;
import java.util.List;

public class DummyCandidate {

  private final int index;

  public int getIndex() {
    return index;
  }

  public DummyCandidate(int index) {
    this.index = index;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null || !(obj instanceof DummyCandidate)) {
      return false;
    }

    DummyCandidate dc = (DummyCandidate) obj;
    return index == dc.index;
  }

  @Override
  public int hashCode() {
    return index;
  }

  public static List<DummyCandidate> generatePopulation(int size) {
    assert size > 0 : "bad size";

    List<DummyCandidate> population = new ArrayList<DummyCandidate>();
    for (int index = 0; index < size; index++) {
      population.add(new DummyCandidate(index));
    }

    return population;
  }
}
