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

package org.apache.mahout.fpm.pfpgrowth;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.StringRecordIterator;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth2.FPGrowthObj;
import org.junit.Test;

import com.google.common.io.Resources;

public final class FPGrowthRetailDataTest2 extends MahoutTestCase {

  @Test
  public void testSpecificCaseFromRetailDataMinSup500() throws IOException {
    FPGrowthObj<String> fp = new FPGrowthObj<String>();
    
    StringRecordIterator it = new StringRecordIterator(new FileLineIterable(Resources.getResource(
      "retail.dat").openStream()), "\\s+");
    int pattern_41_36_39 = 0;
    while (it.hasNext()) {
      Pair<List<String>,Long> next = it.next();
      List<String> items = next.getFirst();
      if (items.contains("41") && items.contains("36") && items.contains("39")) {
        pattern_41_36_39++;
      }
    }
    
    final Map<Set<String>,Long> results = Maps.newHashMap();
    
    Set<String> returnableFeatures = Sets.newHashSet();
    returnableFeatures.add("41");
    returnableFeatures.add("36");
    returnableFeatures.add("39");
    
    fp.generateTopKFrequentPatterns(
      new StringRecordIterator(new FileLineIterable(Resources.getResource("retail.dat").openStream()), "\\s+"),

      fp.generateFList(new StringRecordIterator(new FileLineIterable(Resources.getResource("retail.dat")
          .openStream()), "\\s+"), 500), 500, 1000, returnableFeatures,
      new OutputCollector<String,List<Pair<List<String>,Long>>>() {
        
        @Override
        public void collect(String key, List<Pair<List<String>,Long>> value) {
          
          for (Pair<List<String>,Long> v : value) {
            List<String> l = v.getFirst();
            results.put(Sets.newHashSet(l), v.getSecond());
          }
        }
        
      });
    
    assertEquals(Long.valueOf(pattern_41_36_39), results.get(returnableFeatures));
    
  }

}
