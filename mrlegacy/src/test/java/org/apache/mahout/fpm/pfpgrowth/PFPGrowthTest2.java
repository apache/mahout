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
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class PFPGrowthTest2 extends MahoutTestCase {
  
  private static final Logger log = LoggerFactory.getLogger(PFPGrowthTest.class);
  
  private final Parameters params = new Parameters();

  @Override
  public void setUp() throws Exception {
    super.setUp();
    params.set(PFPGrowth.MIN_SUPPORT, "3");
    params.set(PFPGrowth.MAX_HEAP_SIZE, "4");
    params.set(PFPGrowth.NUM_GROUPS, "2");
    params.set(PFPGrowth.ENCODING, "UTF-8");
    params.set(PFPGrowth.USE_FPG2, "true");
    File inputDir = getTestTempDir("transactions");
    File outputDir = getTestTempDir("frequentpatterns");
    File input = new File(inputDir, "test.txt");
    params.set(PFPGrowth.INPUT, input.getAbsolutePath());
    params.set(PFPGrowth.OUTPUT, outputDir.getAbsolutePath());
    Writer writer = Files.newWriter(input, Charsets.UTF_8);
    try {
      Collection<List<String>> transactions = Lists.newArrayList();
      transactions.add(Arrays.asList("E", "A", "D", "B"));
      transactions.add(Arrays.asList("D", "A", "C", "E", "B"));
      transactions.add(Arrays.asList("C", "A", "B", "E"));
      transactions.add(Arrays.asList("B", "A", "D"));
      transactions.add(Arrays.asList("D", "D", "", "D", "D"));
      transactions.add(Arrays.asList("D", "B"));
      transactions.add(Arrays.asList("A", "D", "E"));
      transactions.add(Arrays.asList("B", "C"));
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
   * Test Parallel FPGrowth on small example data using top-level
   * runPFPGrowth() method
   */ 
  @Test
  public void testStartParallelFPGrowth() throws Exception {
    PFPGrowth.runPFPGrowth(params);

    List<Pair<String,TopKStringPatterns>> frequentPatterns = PFPGrowth.readFrequentPattern(params);

    assertEquals("[(A,([A],5), ([D, A],4), ([B, A],4), ([A, E],4)), "
                 + "(B,([B],6), ([B, D],4), ([B, A],4), ([B, D, A],3)), " 
                 + "(C,([B, C],3)), "
                 + "(D,([D],6), ([D, A],4), ([B, D],4), ([D, A, E],3)), "
                 + "(E,([A, E],4), ([D, A, E],3), ([B, A, E],3))]", frequentPatterns.toString());                                                                 
  }

  /**
   * Test Parallel FPG on small example data, running various stages
   * individually
   */ 
  @Test
  public void testStartParallelFPGrowthInSteps() throws Exception {
    Configuration conf = new Configuration();
    log.info("Starting Parallel Counting Test: {}", params.get(PFPGrowth.MAX_HEAP_SIZE));
    PFPGrowth.startParallelCounting(params, conf);
    log.info("Reading fList Test: {}", params.get(PFPGrowth.MAX_HEAP_SIZE));
    List<Pair<String,Long>> fList = PFPGrowth.readFList(params);
    log.info("{}", fList);
    assertEquals("[(B,6), (D,6), (A,5), (E,4), (C,3)]", fList.toString());
 
    PFPGrowth.saveFList(fList, params, conf);
    int numGroups = params.getInt(PFPGrowth.NUM_GROUPS, 
                                  PFPGrowth.NUM_GROUPS_DEFAULT);
    int maxPerGroup = fList.size() / numGroups;
    if (fList.size() % numGroups != 0) {
      maxPerGroup++;
    }
    params.set(PFPGrowth.MAX_PER_GROUP, Integer.toString(maxPerGroup));

    log.info("Starting Parallel FPGrowth Test: {}", params.get(PFPGrowth.MAX_HEAP_SIZE));
    PFPGrowth.startParallelFPGrowth(params, conf);
    log.info("Starting Pattern Aggregation Test: {}", params.get(PFPGrowth.MAX_HEAP_SIZE));
    PFPGrowth.startAggregating(params, conf);

    List<Pair<String,TopKStringPatterns>> frequentPatterns = PFPGrowth.readFrequentPattern(params);

    assertEquals("[(A,([A],5), ([D, A],4), ([B, A],4), ([A, E],4)), "
                 + "(B,([B],6), ([B, D],4), ([B, A],4), ([B, D, A],3)), " 
                 + "(C,([B, C],3)), "
                 + "(D,([D],6), ([D, A],4), ([B, D],4), ([D, A, E],3)), "
                 + "(E,([A, E],4), ([D, A, E],3), ([B, A, E],3))]", frequentPatterns.toString());                                                                 
  }
}
