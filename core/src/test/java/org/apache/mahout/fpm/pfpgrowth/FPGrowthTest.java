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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.convertors.ContextStatusUpdater;
import org.apache.mahout.fpm.pfpgrowth.convertors.SequenceFileOutputCollector;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.StringOutputConverter;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;
import org.junit.Test;

public final class FPGrowthTest extends MahoutTestCase {

  @Test
  public void testMaxHeapFPGrowth() throws Exception {

    FPGrowth<String> fp = new FPGrowth<String>();

    Collection<Pair<List<String>,Long>> transactions = new ArrayList<Pair<List<String>,Long>>();
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("E", "A", "D", "B"), 1L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("D", "A", "C", "E", "B"), 1L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("C", "A", "B", "E"), 1L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("B", "A", "D"), 1L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("D"), 1L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("D", "B"), 1L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("A", "D", "E"), 1L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("B", "C"), 1L));

    Path path = getTestTempFilePath("fpgrowthTest.dat");
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    SequenceFile.Writer writer =
        new SequenceFile.Writer(fs, conf, path, Text.class, TopKStringPatterns.class);
    fp.generateTopKFrequentPatterns(
        transactions.iterator(),
        fp.generateFList(transactions.iterator(), 3),
        3,
        100,
        new HashSet<String>(),
        new StringOutputConverter(new SequenceFileOutputCollector<Text,TopKStringPatterns>(writer)),
        new ContextStatusUpdater(null));
    writer.close();

    List<Pair<String, TopKStringPatterns>> frequentPatterns = FPGrowth.readFrequentPattern(fs, conf, path);
    assertEquals(
      "[(C,([B, C],3)), "
          + "(E,([A, E],4), ([A, B, E],3), ([A, D, E],3)), "
          + "(A,([A],5), ([A, D],4), ([A, E],4), ([A, B],4), ([A, B, E],3), ([A, D, E],3), ([A, B, D],3)), "
          + "(D,([D],6), ([B, D],4), ([A, D],4), ([A, D, E],3), ([A, B, D],3)), "
          + "(B,([B],6), ([A, B],4), ([B, D],4), ([A, B, D],3), ([A, B, E],3), ([B, C],3))]",
      frequentPatterns.toString());

  }

  /**
   * Trivial test for MAHOUT-355
   */
  @Test
  public void testNoNullPointerExceptionWhenReturnableFeaturesIsNull() throws Exception {

    FPGrowth<String> fp = new FPGrowth<String>();

    Collection<Pair<List<String>,Long>> transactions = new ArrayList<Pair<List<String>,Long>>();
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("E", "A", "D", "B"), 1L));

    OutputCollector<String, List<Pair<List<String>, Long>>> noOutput =
        new OutputCollector<String,List<Pair<List<String>,Long>>>() {
      @Override
      public void collect(String arg0, List<Pair<List<String>, Long>> arg1) { 
      }
    };

    fp.generateTopKFrequentPatterns(
        transactions.iterator(),
        fp.generateFList(transactions.iterator(), 3),
        3,
        100,
        null,
        noOutput,
        new ContextStatusUpdater(null));
  }
}
