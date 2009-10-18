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

package org.apache.mahout.fpm.pfpgrowth.convertors;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FrequentPatternMaxHeap;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.Pattern;

final public class TopKPatternsOutputConvertor<AttributePrimitive> implements
    OutputCollector<Integer, FrequentPatternMaxHeap> {

  private OutputCollector<AttributePrimitive, List<Pair<List<AttributePrimitive>, Long>>> collector = null;

  private Map<Integer, AttributePrimitive> reverseMapping = null;

  public TopKPatternsOutputConvertor(
      OutputCollector<AttributePrimitive, List<Pair<List<AttributePrimitive>, Long>>> collector,
      Map<Integer, AttributePrimitive> reverseMapping) {
    this.collector = collector;
    this.reverseMapping = reverseMapping;
  }

  @Override
  final public void collect(Integer key, FrequentPatternMaxHeap value)
      throws IOException {
    List<Pair<List<AttributePrimitive>, Long>> perAttributePatterns = new ArrayList<Pair<List<AttributePrimitive>, Long>>();
    for (Pattern itemSet : value.getHeap()) {
      List<AttributePrimitive> frequentPattern = new ArrayList<AttributePrimitive>();
      for (int j = 0; j < itemSet.length(); j++) {
        frequentPattern.add(reverseMapping.get(itemSet.getPattern()[j]));
      }
      Pair<List<AttributePrimitive>, Long> returnItemSet = new Pair<List<AttributePrimitive>, Long>(
          frequentPattern, itemSet.getSupport());
      perAttributePatterns.add(returnItemSet);
    }
    collector.collect(reverseMapping.get(key), perAttributePatterns);
  }
}
