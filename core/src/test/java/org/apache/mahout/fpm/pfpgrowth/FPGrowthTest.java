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
import java.io.IOException;
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
import org.apache.mahout.common.Pair;
import org.apache.mahout.fpm.pfpgrowth.convertors.SequenceFileOutputCollector;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.StringOutputConvertor;
import org.apache.mahout.fpm.pfpgrowth.convertors.string.TopKStringPatterns;
import org.apache.mahout.fpm.pfpgrowth.fpgrowth.FPGrowth;

import junit.framework.TestCase;

public class FPGrowthTest extends TestCase {

  public FPGrowthTest(String s) {
    super(s);
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();

  }

  public void testMaxHeapFPGrowth() throws IOException {

    FPGrowth<String> fp = new FPGrowth<String>();

    Collection<List<String>> transactions = new ArrayList<List<String>>();
    transactions.add(Arrays.asList("E", "A", "D", "B"));
    transactions.add(Arrays.asList("D", "A", "C", "E", "B"));
    transactions.add(Arrays.asList("C", "A", "B", "E"));
    transactions.add(Arrays.asList("B", "A", "D"));
    transactions.add(Arrays.asList("D"));
    transactions.add(Arrays.asList("D", "B"));
    transactions.add(Arrays.asList("A", "D", "E"));
    transactions.add(Arrays.asList("B", "C"));

    File tmpDir = new File(System.getProperty("java.io.tmpdir"));
    File tmpLoc = new File(tmpDir, "fpgrowthTest");
    tmpLoc.mkdirs();
    File tmpFile = File.createTempFile("fpgrowthTest", ".dat", tmpLoc);

    Path path = new Path(tmpFile.getAbsolutePath());
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);

    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path,
        Text.class, TopKStringPatterns.class);
    fp.generateTopKFrequentPatterns(transactions.iterator(), fp
        .generateFList(transactions.iterator(), 3), 3, 100,
        new HashSet<String>(), new StringOutputConvertor(
            new SequenceFileOutputCollector<Text, TopKStringPatterns>(writer)
            ));
    writer.close();

    List<Pair<String, TopKStringPatterns>> frequentPatterns = FPGrowth
        .readFrequentPattern(fs, conf, path);
    assertEquals(
        frequentPatterns.toString()
            + " is not equal to [(C,([B, C],3)), (E,([A, E],4), ([D, A, E],3), ([B, A, E],3)), (A,([A],5), ([B, A],4), ([D, A],4), ([A, E],4), ([B, D, A],3), ([D, A, E],3), ([B, A, E],3)), (D,([D],6), ([B, D],4), ([D, A],4), ([B, D, A],3), ([D, A, E],3)), (B,([B],6), ([B, D],4), ([B, A],4), ([B, D, A],3), ([B, A, E],3), ([B, C],3))]",
        "[(C,([B, C],3)), (E,([A, E],4), ([D, A, E],3), ([B, A, E],3)), (A,([A],5), ([B, A],4), ([D, A],4), ([A, E],4), ([B, D, A],3), ([D, A, E],3), ([B, A, E],3)), (D,([D],6), ([B, D],4), ([D, A],4), ([B, D, A],3), ([D, A, E],3)), (B,([B],6), ([B, D],4), ([B, A],4), ([B, D, A],3), ([B, A, E],3), ([B, C],3))]",
        frequentPatterns.toString());

  }
}
