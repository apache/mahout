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
import java.util.Collection;
import java.util.HashMap;
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
import org.apache.mahout.fpm.pfpgrowth.convertors.StatusUpdater;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth2.FPGrowthObj;
import org.junit.Test;

import com.google.common.io.Resources;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class FPGrowthRetailDataTestVs extends MahoutTestCase {

  private static final Logger log = LoggerFactory.getLogger(PFPGrowthRetailDataTestVs.class);

  private static long bestResults(Map<Set<String>, Long> res, Set<String> feats) {
    Long best = res.get(feats);
    if (best != null) {
      return best;
    }
    best = -1L;
    for (Map.Entry<Set<String>, Long> ent : res.entrySet()) { 
      Set<String> r = ent.getKey();
      Long supp = ent.getValue();
      if (supp <= best) {
        continue;
      }
      boolean hasAll = true;
      for (String f : feats) {
        if (!r.contains(f)) {
          hasAll = false;
          break;
        }
      }
      if (hasAll) {
        best = supp;
      }
    }
    return best;
  }

  private static class MapCollector implements OutputCollector<String,List<Pair<List<String>,Long>>> {
    private final Map<Set<String>,Long> results;

    private MapCollector(Map<Set<String>, Long> results) {
      this.results = results;
    }

    @Override
    public void collect(String key, List<Pair<List<String>,Long>> value) {
      for (Pair<List<String>,Long> v : value) {
        List<String> l = v.getFirst();
        results.put(Sets.newHashSet(l), v.getSecond());
        log.info("found pat ["+v.getSecond()+"]: "+ v.getFirst());
      }
    }
  }

  private static class DummyUpdater implements StatusUpdater {
    @Override
    public void update(String status) { }
  }

  @Test
  public void testVsWithRetailData() throws IOException {
    String inputFilename = "retail.dat";
    int minSupport = 500;
    Collection<String> returnableFeatures = Sets.newHashSet();
    
    org.apache.mahout.fpm.pfpgrowth.fpgrowth.
      FPGrowth<String> fp1 = new org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth<String>();

    Map<Set<String>,Long> results1 = Maps.newHashMap();
    
    fp1.generateTopKFrequentPatterns(
      new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename).openStream()), "\\s+"),

      fp1.generateFList(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename)
           .openStream()), "\\s+"), minSupport), minSupport, 100000, 
      returnableFeatures,
      new MapCollector(results1), new DummyUpdater());

    FPGrowthObj<String> fp2 = new FPGrowthObj<String>();
    Map<Set<String>,Long> initialResults2 = Maps.newHashMap();
    fp2.generateTopKFrequentPatterns(
      new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename).openStream()), "\\s+"),

      fp2.generateFList(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename)
           .openStream()), "\\s+"), minSupport), minSupport, 100000,
        Sets.<String>newHashSet(),
      new MapCollector(initialResults2));

    Map<Set<String>, Long> results2;
    if (returnableFeatures.isEmpty()) {
      results2 = initialResults2;
    } else {
      Map<Set<String>, Long> tmpResult = new HashMap<Set<String>, Long>();
      for (Map.Entry<Set<String>, Long> result2 : initialResults2.entrySet()) {
        Set<String> r2feats = result2.getKey();
        boolean hasSome = false;
        for (String rf : returnableFeatures) {
          if (r2feats.contains(rf)) {
            hasSome = true;
            break;
          }
        }
        if (hasSome) {
          tmpResult.put(result2.getKey(), result2.getValue());
        }
      }
      results2 = tmpResult;
    }

    boolean allMatch = hasAll(results1, results2);
    log.info("checked "+results1.size()+" itemsets iterating through #1");

    allMatch &= hasAll(results2, results1);
    log.info("checked "+results2.size()+" itemsets iterating through #2");

    assertTrue("Had mismatches!", allMatch);
  }

  public static boolean hasAll(Map<Set<String>, Long> ref, Map<Set<String>, Long> other) {
    boolean hasAll = true;
    for (Map.Entry<Set<String>, Long> refEnt : ref.entrySet()) {
      Set<String> feats = refEnt.getKey();
      long supp1 = refEnt.getValue();
      long supp2 = bestResults(other, feats);
      if (supp1 != supp2) {
        hasAll = false;
        log.info("mismatch checking results ["+supp1+" vs "+supp2+"]: "+feats);
      }
    }

    return hasAll;
  }

}
