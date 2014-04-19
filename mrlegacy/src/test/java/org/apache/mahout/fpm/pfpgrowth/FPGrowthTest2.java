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

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.convertors.SequenceFileOutputCollector;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.StringOutputConverter;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth2.FPGrowthObj;
import org.junit.Test;

public final class FPGrowthTest2 extends MahoutTestCase {

  @Test
  public void testMaxHeapFPGrowth() throws Exception {

    FPGrowthObj<String> fp = new FPGrowthObj<String>();

    Collection<Pair<List<String>,Long>> transactions = Lists.newArrayList();
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
    FileSystem fs = FileSystem.get(path.toUri(), conf);

    SequenceFile.Writer writer =
        new SequenceFile.Writer(fs, conf, path, Text.class, TopKStringPatterns.class);
    try {
    fp.generateTopKFrequentPatterns(
        transactions.iterator(),
        fp.generateFList(transactions.iterator(), 3),
        3,
        100,
        Sets.<String>newHashSet(),
        new StringOutputConverter(new SequenceFileOutputCollector<Text,TopKStringPatterns>(writer))
    );
    } finally {
      Closeables.close(writer, false);
    }

    List<Pair<String, TopKStringPatterns>> frequentPatterns = FPGrowthObj.readFrequentPattern(conf, path);
    assertEquals(
      "[(C,([B, C],3)), "
          + "(E,([A, E],4), ([A, B, E],3), ([A, D, E],3)), "
          + "(A,([A],5), ([A, D],4), ([A, B],4), ([A, B, D],3)), "
          + "(D,([D],6), ([B, D],4)), "
          + "(B,([B],6))]",
      frequentPatterns.toString());

  }
  
  /**
   * Trivial test for MAHOUT-617
   */
  @Test
  public void testMaxHeapFPGrowthData1() throws Exception {

    FPGrowthObj<String> fp = new FPGrowthObj<String>();

    Collection<Pair<List<String>,Long>> transactions = Lists.newArrayList();
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("X"), 12L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("Y"), 4L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("X", "Y"), 10L));

    Path path = getTestTempFilePath("fpgrowthTestData1.dat");
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    System.out.println(fp.generateFList(transactions.iterator(), 2));
    SequenceFile.Writer writer =
        new SequenceFile.Writer(fs, conf, path, Text.class, TopKStringPatterns.class);
    try {
      fp.generateTopKFrequentPatterns(
          transactions.iterator(),
          fp.generateFList(transactions.iterator(), 2),
          2,
          100,
          Sets.<String>newHashSet(),
          new StringOutputConverter(new SequenceFileOutputCollector<Text,TopKStringPatterns>(writer))
      );
    } finally {
      Closeables.close(writer, false);
    }

    List<Pair<String, TopKStringPatterns>> frequentPatterns = FPGrowthObj.readFrequentPattern(conf, path);
    assertEquals(
      "[(Y,([Y],14), ([X, Y],10)), (X,([X],22))]", frequentPatterns.toString());
  }
  
  /**
   * Trivial test for MAHOUT-617
   */
  @Test
  public void testMaxHeapFPGrowthData2() throws Exception {

    FPGrowthObj<String> fp = new FPGrowthObj<String>();

    Collection<Pair<List<String>,Long>> transactions = Lists.newArrayList();
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("X"), 12L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("Y"), 4L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("X", "Y"), 10L));
    transactions.add(new Pair<List<String>,Long>(Arrays.asList("X", "Y", "Z"), 11L));

    Path path = getTestTempFilePath("fpgrowthTestData2.dat");
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(path.toUri(), conf);
    System.out.println(fp.generateFList(transactions.iterator(), 2));
    SequenceFile.Writer writer =
        new SequenceFile.Writer(fs, conf, path, Text.class, TopKStringPatterns.class);
    try {
      fp.generateTopKFrequentPatterns(
          transactions.iterator(),
          fp.generateFList(transactions.iterator(), 2),
          2,
          100,
          Sets.<String>newHashSet(),
          new StringOutputConverter(new SequenceFileOutputCollector<Text,TopKStringPatterns>(writer))
      );
    } finally {
      Closeables.close(writer, false);
    }

    List<Pair<String, TopKStringPatterns>> frequentPatterns = FPGrowthObj.readFrequentPattern(conf, path);
    assertEquals(
      "[(Z,([X, Y, Z],11)), (Y,([Y],25), ([X, Y],21)), (X,([X],33))]",
      frequentPatterns.toString());
  }

  /**
   * Trivial test for MAHOUT-355
   */
  @Test
  public void testNoNullPointerExceptionWhenReturnableFeaturesIsNull() throws Exception {

    FPGrowthObj<String> fp = new FPGrowthObj<String>();

    Collection<Pair<List<String>,Long>> transactions = Lists.newArrayList();
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
        noOutput
    );
  }
}
