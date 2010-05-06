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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.ToUserPrefsMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.CoRating;
import org.apache.mahout.cf.taste.hadoop.similarity.DistributedUncenteredZeroAssumingCosineSimilarity;
import org.apache.mahout.common.MahoutTestCase;
import org.easymock.IArgumentMatcher;
import org.easymock.classextension.EasyMock;

/**
 * Unit tests for the mappers and reducers in org.apache.mahout.cf.taste.hadoop.similarity
 * Integration test with a mini-file at the end
 *
 */
public final class ItemSimilarityTest extends MahoutTestCase {

  public void testUserPrefsPerItemMapper() throws Exception {
    OutputCollector<LongWritable,LongWritable> output =
        EasyMock.createMock(OutputCollector.class);
    output.collect(new LongWritable(34L), new EntityPrefWritable(12L, 2.3f));
    EasyMock.replay(output);

    new ToUserPrefsMapper().map(new LongWritable(), new Text("12,34,2.3"), output, null);

    EasyMock.verify(output);
  }

  public void testToItemVectorReducer() throws Exception {

    List<EntityPrefWritable> userPrefs = Arrays.asList(
        new EntityPrefWritable(34L, 1.0f), new EntityPrefWritable(56L, 2.0f));

    OutputCollector<LongWritable,EntityPrefWritableArrayWritable> output =
        EasyMock.createMock(OutputCollector.class);

    output.collect(EasyMock.eq(new LongWritable(12L)), equalToUserPrefs(userPrefs));

    EasyMock.replay(output);

    new ToItemVectorReducer().reduce(new LongWritable(12L), userPrefs.iterator(), output, null);

    EasyMock.verify(output);
  }

  static EntityPrefWritableArrayWritable equalToUserPrefs(
      final Collection<EntityPrefWritable> prefsToCheck) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof EntityPrefWritableArrayWritable) {
          EntityPrefWritableArrayWritable userPrefArray =
              (EntityPrefWritableArrayWritable) argument;
          Set<EntityPrefWritable> set = new HashSet<EntityPrefWritable>();
          set.addAll(Arrays.asList(userPrefArray.getPrefs()));

          if (set.size() != prefsToCheck.size()) {
            return false;
          }

          for (EntityPrefWritable prefToCheck : prefsToCheck) {
            if (!set.contains(prefToCheck)) {
              return false;
            }
          }
          return true;
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });

    return null;
  }

  public void testPreferredItemsPerUserMapper() throws Exception {
    OutputCollector<LongWritable,ItemPrefWithItemVectorWeightWritable> output =
        EasyMock.createMock(OutputCollector.class);
    EntityPrefWritableArrayWritable userPrefs = new EntityPrefWritableArrayWritable(
        new EntityPrefWritable[] {
            new EntityPrefWritable(12L, 2.0f),
            new EntityPrefWritable(56L, 3.0f) });

    double weight =
      new DistributedUncenteredZeroAssumingCosineSimilarity().weightOfItemVector(Arrays.asList(2.0f, 3.0f).iterator());

    output.collect(new LongWritable(12L), new ItemPrefWithItemVectorWeightWritable(34L, weight, 2.0f));
    output.collect(new LongWritable(56L), new ItemPrefWithItemVectorWeightWritable(34L, weight, 3.0f));

    JobConf conf = new JobConf();
    conf.set(ItemSimilarityJob.DISTRIBUTED_SIMILARITY_CLASSNAME,
        "org.apache.mahout.cf.taste.hadoop.similarity.DistributedUncenteredZeroAssumingCosineSimilarity");

    EasyMock.replay(output);

    PreferredItemsPerUserMapper mapper = new PreferredItemsPerUserMapper();
    mapper.configure(conf);
    mapper.map(new LongWritable(34L), userPrefs, output, null);

    EasyMock.verify(output);
  }

  public void testPreferredItemsPerUserReducer() throws Exception {

    List<ItemPrefWithItemVectorWeightWritable> itemPrefs =
        Arrays.asList(new ItemPrefWithItemVectorWeightWritable(34L, 5.0, 1.0f),
                      new ItemPrefWithItemVectorWeightWritable(56L, 7.0, 2.0f));

    OutputCollector<LongWritable,ItemPrefWithItemVectorWeightArrayWritable> output =
        EasyMock.createMock(OutputCollector.class);

    output.collect(EasyMock.eq(new LongWritable(12L)), equalToItemPrefs(itemPrefs));

    EasyMock.replay(output);

    new PreferredItemsPerUserReducer().reduce(
        new LongWritable(12L), itemPrefs.iterator(), output, null);

    EasyMock.verify(output);
  }

  static ItemPrefWithItemVectorWeightArrayWritable equalToItemPrefs(
      final Collection<ItemPrefWithItemVectorWeightWritable> prefsToCheck) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof ItemPrefWithItemVectorWeightArrayWritable) {
          ItemPrefWithItemVectorWeightArrayWritable itemPrefArray = (ItemPrefWithItemVectorWeightArrayWritable) argument;
          Collection<ItemPrefWithItemVectorWeightWritable> set = new HashSet<ItemPrefWithItemVectorWeightWritable>();
          set.addAll(Arrays.asList(itemPrefArray.getItemPrefs()));

          if (set.size() != prefsToCheck.size()) {
            return false;
          }

          for (ItemPrefWithItemVectorWeightWritable prefToCheck : prefsToCheck) {
            if (!set.contains(prefToCheck)) {
              return false;
            }
          }
          return true;
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });

    return null;
  }

  public void testCopreferredItemsMapper() throws Exception {
    OutputCollector<ItemPairWritable, CoRating> output =
        EasyMock.createMock(OutputCollector.class);
    ItemPrefWithItemVectorWeightArrayWritable itemPrefs =
        EasyMock.createMock(ItemPrefWithItemVectorWeightArrayWritable.class);

    EasyMock.expect(itemPrefs.getItemPrefs()).andReturn(new ItemPrefWithItemVectorWeightWritable[] {
        new ItemPrefWithItemVectorWeightWritable(34L, 2.0, 1.0f), new ItemPrefWithItemVectorWeightWritable(56L, 3.0, 2.0f),
        new ItemPrefWithItemVectorWeightWritable(78L, 4.0, 3.0f) });

    output.collect(new ItemPairWritable(34L, 56L, 2.0, 3.0), new CoRating(1.0f, 2.0f));
    output.collect(new ItemPairWritable(34L, 78L, 2.0, 4.0), new CoRating(1.0f, 3.0f));
    output.collect(new ItemPairWritable(56L, 78L, 3.0, 4.0), new CoRating(2.0f, 3.0f));

    EasyMock.replay(output, itemPrefs);

    new CopreferredItemsMapper().map(new LongWritable(), itemPrefs, output, null);

    EasyMock.verify(output, itemPrefs);
  }

  public void testSimilarityReducer() throws Exception {
    OutputCollector<EntityEntityWritable,DoubleWritable> output =
        EasyMock.createMock(OutputCollector.class);

    JobConf conf = new JobConf();
    conf.set(ItemSimilarityJob.DISTRIBUTED_SIMILARITY_CLASSNAME,
        "org.apache.mahout.cf.taste.hadoop.similarity.DistributedUncenteredZeroAssumingCosineSimilarity");

    output.collect(new EntityEntityWritable(12L, 34L), new DoubleWritable(0.5));

    EasyMock.replay(output);

    SimilarityReducer reducer = new SimilarityReducer();
    reducer.configure(conf);
    reducer.reduce(new ItemPairWritable(12L, 34L, 2.0, 10.0), Arrays.asList(new CoRating(2.5f, 2.0f),
            new CoRating(2.0f, 2.5f)).iterator(), output, null);

    EasyMock.verify(output);
  }

  public void testCompleteJob() throws Exception {

    String tmpDirProp = System.getProperty("java.io.tmpdir");
    if (!tmpDirProp.endsWith("/")) {
      tmpDirProp += "/";
    }
    String tmpDirPath = tmpDirProp + ItemSimilarityTest.class.getCanonicalName();
    File tmpDir = new File(tmpDirPath);

    try {
      if (tmpDir.exists()) {
        recursiveDelete(tmpDir);
      }
      tmpDir.mkdirs();

      /* user-item-matrix

                   Game   Mouse   PC    Disk
           Jane     -       1      2      -
           Paul     1       -      1      -
           Fred     -       -      -      1
       */

      BufferedWriter writer = new BufferedWriter(new FileWriter(tmpDirPath+"/prefs.txt"));
      try {
        writer.write("1,2,1\n" +
                     "1,3,2\n" +
                     "2,1,1\n" +
                     "2,3,1\n" +
                     "3,4,1\n");
      } finally {
        writer.close();
      }

      ItemSimilarityJob similarityJob = new ItemSimilarityJob();

      Configuration conf = new Configuration();
      conf.set("mapred.input.dir", tmpDirPath+"/prefs.txt");
      conf.set("mapred.output.dir", tmpDirPath+"/output");
      conf.set("mapred.output.compress", Boolean.FALSE.toString());

      similarityJob.setConf(conf);

      similarityJob.run(new String[] { "--tempDir", tmpDirPath+"/tmp", "--similarityClassname",
          "org.apache.mahout.cf.taste.hadoop.similarity.DistributedUncenteredZeroAssumingCosineSimilarity"});

      String filePath = tmpDirPath+"/output/part-00000";
      BufferedReader reader = new BufferedReader(new FileReader(filePath));

      String line;
      int currentLine = 1;
      while ( (line = reader.readLine()) != null) {

        String[] tokens = line.split("\t");

        long itemAID = Long.parseLong(tokens[0]);
        long itemBID = Long.parseLong(tokens[1]);
        double similarity = Double.parseDouble(tokens[2]);

        if (currentLine == 1) {
          assertEquals(1L, itemAID);
          assertEquals(3L, itemBID);
          assertEquals(0.45, similarity, 0.01);
        }

        if (currentLine == 2) {
          assertEquals(2L, itemAID);
          assertEquals(3L, itemBID);
          assertEquals(0.89, similarity, 0.01);
        }

        currentLine++;
      }

      int linesWritten = currentLine-1;
      assertEquals(2, linesWritten);

    } finally {
      recursiveDelete(tmpDir);
    }
  }

  static void recursiveDelete(File fileOrDir) {
    if (fileOrDir.isDirectory()) {
      for (File innerFile : fileOrDir.listFiles()) {
        recursiveDelete(innerFile);
      }
    }
    fileOrDir.delete();
  }

}
