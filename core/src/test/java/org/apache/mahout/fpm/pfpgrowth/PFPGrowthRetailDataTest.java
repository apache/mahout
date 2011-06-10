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
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.StringRecordIterator;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Resources;

public class PFPGrowthRetailDataTest extends MahoutTestCase {
  
  private final Parameters params = new Parameters();
  private static final Logger log = LoggerFactory.getLogger(PFPGrowthRetailDataTest.class);
  
  @Override
  public void setUp() throws Exception {
    super.setUp();
    params.set(PFPGrowth.MIN_SUPPORT, "100");
    params.set(PFPGrowth.MAX_HEAPSIZE, "10000");
    params.set(PFPGrowth.NUM_GROUPS, "50");
    params.set(PFPGrowth.ENCODING, "UTF-8");
    File inputDir = getTestTempDir("transactions");
    File outputDir = getTestTempDir("frequentpatterns");
    File input = new File(inputDir, "test.txt");
    params.set(PFPGrowth.INPUT, input.getAbsolutePath());
    params.set(PFPGrowth.OUTPUT, outputDir.getAbsolutePath());
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
      Closeables.closeQuietly(writer);
    }
  }
  
  @Test
  public void testRetailDataMinSup100() throws Exception {
    StringRecordIterator it = new StringRecordIterator(new FileLineIterable(Resources.getResource(
      "retail_results_with_min_sup_100.dat").openStream()), "\\s+");
    
    Map<Set<String>,Long> expectedResults = Maps.newHashMap();
    while (it.hasNext()) {
      Pair<List<String>,Long> next = it.next();
      List<String> items = Lists.newArrayList(next.getFirst());
      String supportString = items.remove(items.size() - 1);
      Long support = Long.parseLong(supportString.substring(1, supportString.length() - 1));
      expectedResults.put(new HashSet<String>(items), support);
    }
    
    log.info("Starting Parallel Counting Test: {}", params.get(PFPGrowth.MAX_HEAPSIZE));
    PFPGrowth.startParallelCounting(params);
    log.info("Starting Grouping Test: {}", params.get(PFPGrowth.MAX_HEAPSIZE));
    PFPGrowth.startGroupingItems(params);
    log.info("Starting Parallel FPGrowth Test: {}", params.get(PFPGrowth.MAX_HEAPSIZE));
    PFPGrowth.startGroupingItems(params);
    PFPGrowth.startTransactionSorting(params);
    PFPGrowth.startParallelFPGrowth(params);
    log.info("Starting Pattern Aggregation Test: {}", params.get(PFPGrowth.MAX_HEAPSIZE));
    PFPGrowth.startAggregating(params);
    List<Pair<String,TopKStringPatterns>> frequentPatterns = PFPGrowth.readFrequentPattern(params);
    
    Map<Set<String>,Long> results = Maps.newHashMap();
    for (Pair<String,TopKStringPatterns> topK : frequentPatterns) {
      Iterator<Pair<List<String>,Long>> topKIt = topK.getSecond().iterator();
      while (topKIt.hasNext()) {
        Pair<List<String>,Long> entry = topKIt.next();
        results.put(new HashSet<String>(entry.getFirst()), entry.getSecond());
      }
    }
    
    for (Entry<Set<String>,Long> entry : results.entrySet()) {
      Set<String> key = entry.getKey();
      if (expectedResults.get(key) == null) {
        System.out.println("missing: " + key);
      } else {
        if (!expectedResults.get(key).equals(results.get(entry.getKey()))) {
          System.out.println("invalid: " + key + ", expected: " + expectedResults.get(key) + ", got: "
                             + results.get(entry.getKey()));
        }
      }
    }
    
    for (Entry<Set<String>,Long> entry : expectedResults.entrySet()) {
      Set<String> key = entry.getKey();
      if (results.get(key) == null) {
        System.out.println("missing: " + key);
      }
    }
    assertEquals(expectedResults.size(), results.size());
  }
}
