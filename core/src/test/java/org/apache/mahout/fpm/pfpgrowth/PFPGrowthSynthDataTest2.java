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

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.Parameters;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.StringRecordIterator;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth2.FPGrowthObj;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.io.Resources;

public class PFPGrowthSynthDataTest2 extends MahoutTestCase {
  
  private final Parameters params = new Parameters();
  private static final Logger log = LoggerFactory.getLogger(PFPGrowthSynthDataTest2.class);
  
  @Override
  public void setUp() throws Exception {
    super.setUp();
    params.set(PFPGrowth.MIN_SUPPORT, "100");
    params.set(PFPGrowth.MAX_HEAP_SIZE, "10000");
    params.set(PFPGrowth.NUM_GROUPS, "50");
    params.set(PFPGrowth.ENCODING, "UTF-8");
    params.set(PFPGrowth.USE_FPG2, "true");
    params.set(PFPGrowth.SPLIT_PATTERN, " ");
    File inputDir = getTestTempDir("transactions");
    File outputDir = getTestTempDir("frequentpatterns");
    File input = new File(inputDir, "synth_test.txt");
    params.set(PFPGrowth.INPUT, input.getAbsolutePath());
    params.set(PFPGrowth.OUTPUT, outputDir.getAbsolutePath());
    Writer writer = Files.newWriter(input, Charsets.UTF_8);
    try {
      StringRecordIterator it = new StringRecordIterator(new FileLineIterable(Resources.getResource(
        "FPGsynth.dat").openStream()), "\\s+");
      Collection<List<String>> transactions = Lists.newArrayList();
      
      while (it.hasNext()) {
        Pair<List<String>,Long> next = it.next();
        transactions.add(next.getFirst());
      }
      
      for (List<String> transaction : transactions) {
        String sep = "";
        for (String item : transaction) {
          writer.write(sep + item);
          sep = " ";
        }
        writer.write("\n");
      }
      
    } finally {
      Closeables.close(writer, false);
    }
  }

  @Test
  public void testVsSequential() throws Exception {

    Map<Set<String>,Long> parallelResult = Maps.newHashMap();

    PFPGrowth.runPFPGrowth(params);
    List<Pair<String,TopKStringPatterns>> tmpParallel = PFPGrowth.readFrequentPattern(params);
    
    for (Pair<String,TopKStringPatterns> topK : tmpParallel) {
      Iterator<Pair<List<String>,Long>> topKIt = topK.getSecond().iterator();
      while (topKIt.hasNext()) {
        Pair<List<String>,Long> entry = topKIt.next();
        parallelResult.put(Sets.newHashSet(entry.getFirst()), entry.getSecond());
      }
    }

    String inputFilename= "FPGsynth.dat";
    int minSupport= 100;

    final Map<Set<String>,Long> seqResult = Maps.newHashMap();
    
    FPGrowthObj<String> fpSeq = new FPGrowthObj<String>();
    fpSeq.generateTopKFrequentPatterns(
      new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename).openStream()), "\\s+"),

      fpSeq.generateFList(new StringRecordIterator(new FileLineIterable(Resources.getResource(inputFilename)
           .openStream()), "\\s+"), minSupport), minSupport, 1000000, 
      null,
      new OutputCollector<String,List<Pair<List<String>,Long>>>() {
        
        @Override
        public void collect(String key, List<Pair<List<String>,Long>> value) {
          
          for (Pair<List<String>,Long> v : value) {
            List<String> l = v.getFirst();
            seqResult.put(Sets.newHashSet(l), v.getSecond());
          }
        }
        
      });

    for (Entry<Set<String>,Long> entry : parallelResult.entrySet()) {
      Set<String> key = entry.getKey();
      if (seqResult.get(key) == null) {
        log.info("spurious (1): " + key+ " with " +entry.getValue());
      } else {
        if (seqResult.get(key).equals(parallelResult.get(entry.getKey()))) {
          log.info("matched (1): " + key + ", with: " + seqResult.get(key));
        } else {
          log.info("invalid (1): " + key + ", expected: " + seqResult.get(key) + ", got: "
                       + parallelResult.get(entry.getKey()));
        }
      }
    }
  
    for (Entry<Set<String>,Long> entry : seqResult.entrySet()) {
      Set<String> key = entry.getKey();
      if (parallelResult.get(key) == null) {
        log.info("missing (1): " + key+ " with " +entry.getValue());
      }
    }
    assertEquals(seqResult.size(), parallelResult.size());
  }

}
