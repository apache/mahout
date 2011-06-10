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

package org.apache.mahout.ga.watchmaker.cd.utils;

import com.google.common.collect.Maps;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;

import java.util.Map;

public final class RandomRuleResults {

  private static final Map<Integer, CDFitness> results = Maps.newHashMap();

  private RandomRuleResults() {
  }

  public static synchronized void addResult(int ruleid, CDFitness fit) {
    CDFitness f = results.get(ruleid);
    if (f == null) {
      f = new CDFitness(fit);
    } else {
      f.add(fit);
    }
    
    results.put(ruleid, f);
  }

  public static CDFitness getResult(int ruleid) {
    return results.get(ruleid);
  }
}
