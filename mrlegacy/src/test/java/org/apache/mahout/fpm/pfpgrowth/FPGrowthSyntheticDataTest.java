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
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth2.FPGrowthObj;
import org.junit.Test;

import com.google.common.io.Resources;

public final class FPGrowthSyntheticDataTest extends MahoutTestCase {

  @Test
  public void testSpecificCasesFromSynthData() throws IOException {
    FPGrowthObj<String> fp = new FPGrowthObj<String>();
    
    String inputFilename = "FPGsynth.dat";

    StringRecordIterator it =
        new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename).openStream()), "\\s+");
    int patternCnt_10_13_1669 = 0;
    int patternCnt_10_13 = 0;
    while (it.hasNext()) {
      Pair<List<String>,Long> next = it.next();
      List<String> items = next.getFirst();
      if (items.contains("10") && items.contains("13")) {
        patternCnt_10_13++;
        if (items.contains("1669")) {
          patternCnt_10_13_1669++;
        }
      }
    }

    int minSupport = 50;
    if (patternCnt_10_13_1669 < minSupport) {
      throw new IllegalStateException("the test is broken or data is missing ("
                                          + patternCnt_10_13_1669 + ", "
                                          + patternCnt_10_13 + ')');
    }

    final Map<Set<String>,Long> results = Maps.newHashMap();
    
    Set<String> features_10_13 = Sets.newHashSet();
    features_10_13.add("10");
    features_10_13.add("13");

    Set<String> returnableFeatures = Sets.newHashSet();
    returnableFeatures.add("10");
    returnableFeatures.add("13");
    returnableFeatures.add("1669");
    
    fp.generateTopKFrequentPatterns(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename).openStream()), "\\s+"),

                                    fp.generateFList(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename)
                                                                                                   .openStream()), "\\s+"), minSupport), minSupport, 100000, 
                                    returnableFeatures,
                                    new OutputCollector<String,List<Pair<List<String>,Long>>>() {
        
                                      @Override
                                        public void collect(String key, List<Pair<List<String>,Long>> value) {
          
                                        for (Pair<List<String>,Long> v : value) {
                                          List<String> l = v.getFirst();
                                          results.put(Sets.newHashSet(l), v.getSecond());
                                          System.out.println("found pat ["+v.getSecond()+"]: "+ v.getFirst());
                                        }
                                      }
        
                                    });

    assertEquals(patternCnt_10_13, highestSupport(results, features_10_13));
    assertEquals(patternCnt_10_13_1669, highestSupport(results, returnableFeatures));
    
  }

  private static long highestSupport(Map<Set<String>, Long> res, Set<String> feats) {
    Long best= res.get(feats);
    if (best != null) {
      return best;
    }
    best = -1L;
    for (Map.Entry<Set<String>, Long> ent : res.entrySet()) {
      Set<String> r= ent.getKey();
      Long supp= ent.getValue();
      if (supp <= best) {
        continue;
      }
      boolean hasAll= true;
      for (String f : feats) {
        if (!r.contains(f)) {
          hasAll= false;
          break;
        }
      }
      if (hasAll) {
        best = supp;
      }
    }
    return best;
  }

  @Test
  public void testVsWithSynthData() throws IOException {
    Collection<String> returnableFeatures = Sets.newHashSet();

    // not limiting features (or including too many) can cause
    // the test to run a very long time
    returnableFeatures.add("10");
    returnableFeatures.add("13");
    //    returnableFeatures.add("1669");
    
    FPGrowth<String> fp1 = new FPGrowth<String>();

    final Map<Set<String>,Long> results1 = Maps.newHashMap();

    String inputFilename = "FPGsynth.dat";
    int minSupport = 100;
    fp1.generateTopKFrequentPatterns(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename).openStream()), "\\s+"),

                                     fp1.generateFList(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename)
                                                                                                     .openStream()), "\\s+"), minSupport), minSupport, 1000000, 
                                     returnableFeatures,
                                     new OutputCollector<String,List<Pair<List<String>,Long>>>() {
        
                                       @Override
                                         public void collect(String key, List<Pair<List<String>,Long>> value) {
          
                                         for (Pair<List<String>,Long> v : value) {
                                           List<String> l = v.getFirst();
                                           results1.put(Sets.newHashSet(l), v.getSecond());
                                           System.out.println("found pat ["+v.getSecond()+"]: "+ v.getFirst());
                                         }
                                       }
        
                                     }, new StatusUpdater() {
        
                                         @Override
                                           public void update(String status) {}
                                       });

    FPGrowthObj<String> fp2 = new FPGrowthObj<String>();
    final Map<Set<String>,Long> initialResults2 = Maps.newHashMap();
    fp2.generateTopKFrequentPatterns(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename).openStream()), "\\s+"),

                                     fp2.generateFList(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename)
                                                                                                     .openStream()), "\\s+"), minSupport), minSupport, 1000000,
                                     Sets.<String>newHashSet(),
                                     new OutputCollector<String,List<Pair<List<String>,Long>>>() {
        
                                       @Override
                                         public void collect(String key, List<Pair<List<String>,Long>> value) {
          
                                         for (Pair<List<String>,Long> v : value) {
                                           List<String> l = v.getFirst();
                                           initialResults2.put(Sets.newHashSet(l), v.getSecond());
                                           System.out.println("found pat ["+v.getSecond()+"]: "+ v.getFirst());
                                         }
                                       }
        
                                     });

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

    boolean allMatch = true;
    int itemsetsChecked= 0;
    for (Map.Entry<Set<String>, Long> result1 : results1.entrySet()) {
      itemsetsChecked++;
      Set<String> feats= result1.getKey();
      long supp1= result1.getValue();
      long supp2= highestSupport(results2, feats);
      if (supp1 != supp2) {
        allMatch= false;
        System.out.println("mismatch checking results1 [ "+supp1+" vs "+supp2+"]: "+feats);
      }
    }
    System.out.println("checked "+itemsetsChecked+" itemsets iterating through #1");

    itemsetsChecked= 0;
    for (Map.Entry<Set<String>, Long> result2 : results2.entrySet()) { 
      itemsetsChecked++;
      Set<String> feats= result2.getKey();
      long supp2= result2.getValue();
      long supp1= highestSupport(results1, feats);
      if (supp1 != supp2) {
        allMatch= false;
        System.out.println("mismatch checking results2 [ "+supp1+" vs "+supp2+"]: "+feats);
      }
    }
    System.out.println("checked "+itemsetsChecked+" itemsets iterating through #2");

    assertTrue("Had mismatches!", allMatch);
  }

}
