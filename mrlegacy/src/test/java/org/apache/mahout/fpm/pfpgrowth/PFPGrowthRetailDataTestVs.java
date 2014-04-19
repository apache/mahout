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

import java.io.File;
import java.io.Writer;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import com.google.common.collect.Maps;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.StringRecordIterator;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.junit.Test;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import com.google.common.io.Resources;

public final class PFPGrowthRetailDataTestVs extends MahoutTestCase {

  private final Parameters paramsImpl1 = new Parameters();
  private final Parameters paramsImpl2 = new Parameters();

  @Override
  public void setUp() throws Exception {
    super.setUp();

    File inputDir = getTestTempDir("transactions");
    File input = new File(inputDir, "test.txt");

    paramsImpl1.set(PFPGrowth.MIN_SUPPORT, "100");
    paramsImpl1.set(PFPGrowth.MAX_HEAP_SIZE, "10000");
    paramsImpl1.set(PFPGrowth.NUM_GROUPS, "50");
    paramsImpl1.set(PFPGrowth.ENCODING, "UTF-8");
    paramsImpl1.set(PFPGrowth.INPUT, input.getAbsolutePath());

    paramsImpl2.set(PFPGrowth.MIN_SUPPORT, "100");
    paramsImpl2.set(PFPGrowth.MAX_HEAP_SIZE, "10000");
    paramsImpl2.set(PFPGrowth.NUM_GROUPS, "50");
    paramsImpl2.set(PFPGrowth.ENCODING, "UTF-8");
    paramsImpl2.set(PFPGrowth.INPUT, input.getAbsolutePath());
    paramsImpl2.set(PFPGrowth.USE_FPG2, "true");

    File outputDir1 = getTestTempDir("frequentpatterns1");
    paramsImpl1.set(PFPGrowth.OUTPUT, outputDir1.getAbsolutePath());

    File outputDir2 = getTestTempDir("frequentpatterns2");
    paramsImpl2.set(PFPGrowth.OUTPUT, outputDir2.getAbsolutePath());

    Writer writer = Files.newWriter(input, Charsets.UTF_8);
    try {
      StringRecordIterator it = new StringRecordIterator(new FileLineIterable(Resources.getResource(
        "retail.dat").openStream()), "\\s+");
      Collection<List<String>> transactions = Lists.newArrayList();
      
      while (it.hasNext()) {
        Pair<List<String>,Long> next = it.next();
        transactions.add(next.getFirst());
      }
      
      for (List<String> transaction : transactions) {
        String sep = "";
        for (String item : transaction) {
          writer.write(sep + item);
          sep = ",";
        }
        writer.write("\n");
      }
      
    } finally {
      Closeables.close(writer, false);
    }
  }
  
   
  /**
   * Test Parallel FPGrowth on retail data using top-level runPFPGrowth() method
   */ 
  @Test
  public void testParallelRetailVs() throws Exception {

    PFPGrowth.runPFPGrowth(paramsImpl1);
    List<Pair<String,TopKStringPatterns>> frequentPatterns1 = PFPGrowth.readFrequentPattern(paramsImpl1);
    
    Map<Set<String>,Long> results1 = Maps.newHashMap();
    for (Pair<String,TopKStringPatterns> topK : frequentPatterns1) {
      Iterator<Pair<List<String>,Long>> topKIt = topK.getSecond().iterator();
      while (topKIt.hasNext()) {
        Pair<List<String>,Long> entry = topKIt.next();
        results1.put(Sets.newHashSet(entry.getFirst()), entry.getSecond());
      }
    }
  
    PFPGrowth.runPFPGrowth(paramsImpl2);
    List<Pair<String,TopKStringPatterns>> frequentPatterns2 = PFPGrowth.readFrequentPattern(paramsImpl2);
  
    Map<Set<String>,Long> results2 = Maps.newHashMap();
    for (Pair<String,TopKStringPatterns> topK : frequentPatterns2) {
      Iterator<Pair<List<String>,Long>> topKIt = topK.getSecond().iterator();
      while (topKIt.hasNext()) {
        Pair<List<String>,Long> entry = topKIt.next();
        results2.put(Sets.newHashSet(entry.getFirst()), entry.getSecond());
      }
    }
  
    for (Entry<Set<String>,Long> entry : results1.entrySet()) {
      Set<String> key = entry.getKey();
      if (results2.get(key) == null) {
        System.out.println("spurious (1): " + key+ " with " +entry.getValue());
      } else {
        if (!results2.get(key).equals(results1.get(entry.getKey()))) {
          System.out.println("invalid (1): " + key + ", expected: " + results2.get(key) + ", got: "
                             + results1.get(entry.getKey()));
        } else {
          System.out.println("matched (1): " + key + ", with: " + results2.get(key));
        }
      }
    }
  
    for (Entry<Set<String>,Long> entry : results2.entrySet()) {
      Set<String> key = entry.getKey();
      if (results1.get(key) == null) {
        System.out.println("missing (1): " + key+ " with " +entry.getValue());
      }
    }
    assertEquals(results2.size(), results1.size());
  }

}
