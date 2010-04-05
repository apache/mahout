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
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.OutputCollector;
import org.easymock.classextension.EasyMock;
import org.easymock.IArgumentMatcher;

import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.ToUserPrefsMapper;
import org.apache.mahout.common.MahoutTestCase;

/**
 * Unit tests for the mappers and reducers in org.apache.mahout.cf.taste.hadoop.similarity
 * Integration test with a mini-file at the end
 *
 */
public class ItemSimilarityTest extends MahoutTestCase {


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
    OutputCollector<LongWritable,ItemPrefWithLengthWritable> output =
        EasyMock.createMock(OutputCollector.class);
    EntityPrefWritableArrayWritable userPrefs =
        EasyMock.createMock(EntityPrefWritableArrayWritable.class);

    EasyMock.expect(userPrefs.getPrefs()).andReturn(
        new EntityPrefWritable[] {
            new EntityPrefWritable(12L, 2.0f),
            new EntityPrefWritable(56L, 3.0f) });

    double length = Math.sqrt(Math.pow(2.0f, 2) + Math.pow(3.0f, 2));

    output.collect(new LongWritable(12L), new ItemPrefWithLengthWritable(34L, length, 2.0f));
    output.collect(new LongWritable(56L), new ItemPrefWithLengthWritable(34L, length, 3.0f));

    EasyMock.replay(output, userPrefs);

    new PreferredItemsPerUserMapper().map(new LongWritable(34L), userPrefs, output, null);

    EasyMock.verify(output, userPrefs);
  }

  public void testPreferredItemsPerUserReducer() throws Exception {

    List<ItemPrefWithLengthWritable> itemPrefs =
        Arrays.asList(new ItemPrefWithLengthWritable(34L, 5.0, 1.0f),
                      new ItemPrefWithLengthWritable(56L, 7.0, 2.0f));

    OutputCollector<LongWritable,ItemPrefWithLengthArrayWritable> output =
        EasyMock.createMock(OutputCollector.class);

    output.collect(EasyMock.eq(new LongWritable(12L)), equalToItemPrefs(itemPrefs));

    EasyMock.replay(output);

    new PreferredItemsPerUserReducer().reduce(
        new LongWritable(12L), itemPrefs.iterator(), output, null);

    EasyMock.verify(output);
  }

  static ItemPrefWithLengthArrayWritable equalToItemPrefs(
      final Collection<ItemPrefWithLengthWritable> prefsToCheck) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof ItemPrefWithLengthArrayWritable) {
          ItemPrefWithLengthArrayWritable itemPrefArray = (ItemPrefWithLengthArrayWritable) argument;
          Collection<ItemPrefWithLengthWritable> set = new HashSet<ItemPrefWithLengthWritable>();
          set.addAll(Arrays.asList(itemPrefArray.getItemPrefs()));

          if (set.size() != prefsToCheck.size()) {
            return false;
          }

          for (ItemPrefWithLengthWritable prefToCheck : prefsToCheck) {
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
    OutputCollector<ItemPairWritable,FloatWritable> output =
        EasyMock.createMock(OutputCollector.class);
    ItemPrefWithLengthArrayWritable itemPrefs =
        EasyMock.createMock(ItemPrefWithLengthArrayWritable.class);

    EasyMock.expect(itemPrefs.getItemPrefs()).andReturn(new ItemPrefWithLengthWritable[] {
        new ItemPrefWithLengthWritable(34L, 2.0, 1.0f), new ItemPrefWithLengthWritable(56L, 3.0, 2.0f),
        new ItemPrefWithLengthWritable(78L, 4.0, 3.0f) });

    output.collect(new ItemPairWritable(34L, 56L, 6.0), new FloatWritable(2.0f));
    output.collect(new ItemPairWritable(34L, 78L, 8.0), new FloatWritable(3.0f));
    output.collect(new ItemPairWritable(56L, 78L, 12.0), new FloatWritable(6.0f));

    EasyMock.replay(output, itemPrefs);

    new CopreferredItemsMapper().map(new LongWritable(), itemPrefs, output, null);

    EasyMock.verify(output, itemPrefs);
  }

  public void testCosineSimilarityReducer() throws Exception {
    OutputCollector<EntityEntityWritable,DoubleWritable> output =
        EasyMock.createMock(OutputCollector.class);

    output.collect(new EntityEntityWritable(12L, 34L), new DoubleWritable(0.5d));

    EasyMock.replay(output);

    new CosineSimilarityReducer().reduce(new ItemPairWritable(12L, 34L, 20.0),
        Arrays.asList(new FloatWritable(5.0f),
                      new FloatWritable(5.0f)).iterator(), output, null);

    EasyMock.verify(output);
  }

  public void testCompleteJob() throws Exception {

    String tmpDirPath = System.getProperty("java.io.tmpdir") +
          ItemSimilarityTest.class.getCanonicalName();
    File tmpDir = new File(tmpDirPath);

    try {
      if (tmpDir.exists()) {
        recursiveDelete(tmpDir);
      } else {
        tmpDir.mkdirs();
      }

      /* user-item-matrix

                   Game   Mouse   PC    Disk
           Jane     0       1      2      0
           Paul     1       0      1      0
           Fred     0       0      0      1
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

      similarityJob.run(new String[] { "--tempDir", tmpDirPath+"/tmp"});

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
