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

package org.apache.mahout.cf.taste.hadoop.similarity;

import static org.easymock.EasyMock.eq;
import static org.easymock.EasyMock.expect;
import static org.easymock.classextension.EasyMock.createMock;
import static org.easymock.classextension.EasyMock.replay;
import static org.easymock.classextension.EasyMock.verify;

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
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.ItemItemWritable;
import org.apache.mahout.cf.taste.hadoop.ItemWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.CopreferredItemsMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.item.CosineSimilarityReducer;
import org.apache.mahout.cf.taste.hadoop.similarity.item.ItemSimilarityJob;
import org.apache.mahout.cf.taste.hadoop.similarity.item.PreferredItemsPerUserMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.item.PreferredItemsPerUserReducer;
import org.apache.mahout.cf.taste.hadoop.similarity.item.ToItemVectorReducer;
import org.apache.mahout.cf.taste.hadoop.similarity.item.UserPrefsPerItemMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPairWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthArrayWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.UserPrefArrayWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.UserPrefWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.UserWritable;
import org.apache.mahout.common.MahoutTestCase;
import org.easymock.IArgumentMatcher;
import org.easymock.classextension.EasyMock;

/**
 * Unit tests for the mappers and reducers in org.apache.mahout.cf.taste.hadoop.similarity
 * Integration test with a mini-file at the end
 *
 */
@SuppressWarnings("unchecked")
public class ItemSimilarityTest extends MahoutTestCase {


  public void testUserPrefsPerItemMapper() throws Exception {
    Mapper.Context ctx = createMock(Mapper.Context.class);
    ctx.write(new ItemWritable(34l), new UserPrefWritable(12l, 2.3f));
    replay(ctx);

    new UserPrefsPerItemMapper().map(new LongWritable(), new Text("12,34,2.3"), ctx);

    verify(ctx);
  }

  public void testToItemVectorReducer() throws Exception {

    List<UserPrefWritable> userPrefs = Arrays.asList(new UserPrefWritable(34l, 1f), new UserPrefWritable(56l, 2f));

    Reducer.Context ctx = createMock(Reducer.Context.class);

    ctx.write(eq(new ItemWritable(12l)), equalToUserPrefs(userPrefs));

    replay(ctx);

    new ToItemVectorReducer().reduce(new ItemWritable(12l), userPrefs, ctx);

    verify(ctx);
  }

  static UserPrefArrayWritable equalToUserPrefs(final Collection<UserPrefWritable> prefsToCheck) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof UserPrefArrayWritable) {
          UserPrefArrayWritable userPrefArray = (UserPrefArrayWritable) argument;
          Set<UserPrefWritable> set = new HashSet<UserPrefWritable>();
          for (UserPrefWritable userPref : userPrefArray.getUserPrefs()) {
            set.add(userPref);
          }

          if (set.size() != prefsToCheck.size()) {
            return false;
          }

          for (UserPrefWritable prefToCheck : prefsToCheck) {
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
    Mapper.Context ctx = createMock(Mapper.Context.class);
    UserPrefArrayWritable userPrefs = createMock(UserPrefArrayWritable.class);

    expect(userPrefs.getUserPrefs())
        .andReturn(new UserPrefWritable[] { new UserPrefWritable(12l, 2f), new UserPrefWritable(56l, 3f) });

    double length = Math.sqrt(Math.pow(2f, 2) + Math.pow(3f, 2));

    ctx.write(new UserWritable(12l), new ItemPrefWithLengthWritable(34l, length, 2f));
    ctx.write(new UserWritable(56l), new ItemPrefWithLengthWritable(34l, length, 3f));

    replay(ctx, userPrefs);

    new PreferredItemsPerUserMapper().map(new ItemWritable(34l), userPrefs, ctx);

    verify(ctx, userPrefs);
  }

  public void testPreferredItemsPerUserReducer() throws Exception {

    List<ItemPrefWithLengthWritable> itemPrefs =
        Arrays.asList(new ItemPrefWithLengthWritable(34l, 5d, 1f), new ItemPrefWithLengthWritable(56l, 7d, 2f));

    Reducer.Context ctx = createMock(Reducer.Context.class);

    ctx.write(eq(new UserWritable(12l)), equalToItemPrefs(itemPrefs));

    replay(ctx);

    new PreferredItemsPerUserReducer().reduce(new UserWritable(12l), itemPrefs, ctx);

    verify(ctx);
  }

  static ItemPrefWithLengthArrayWritable equalToItemPrefs(final Collection<ItemPrefWithLengthWritable> prefsToCheck) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof ItemPrefWithLengthArrayWritable) {
          ItemPrefWithLengthArrayWritable itemPrefArray = (ItemPrefWithLengthArrayWritable) argument;
          Set<ItemPrefWithLengthWritable> set = new HashSet<ItemPrefWithLengthWritable>();
          for (ItemPrefWithLengthWritable itemPref : itemPrefArray.getItemPrefs()) {
            set.add(itemPref);
          }

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
    Mapper.Context ctx = createMock(Mapper.Context.class);
    ItemPrefWithLengthArrayWritable itemPrefs = createMock(ItemPrefWithLengthArrayWritable.class);

    expect(itemPrefs.getItemPrefs()).andReturn(new ItemPrefWithLengthWritable[] {
        new ItemPrefWithLengthWritable(34l, 2d, 1f), new ItemPrefWithLengthWritable(56l, 3d, 2f),
        new ItemPrefWithLengthWritable(78l, 4d, 3f) });

    ctx.write(new ItemPairWritable(34l, 56l, 6d), new FloatWritable(2f));
    ctx.write(new ItemPairWritable(34l, 78l, 8d), new FloatWritable(3f));
    ctx.write(new ItemPairWritable(56l, 78l, 12d), new FloatWritable(6f));

    replay(ctx, itemPrefs);

    new CopreferredItemsMapper().map(new UserWritable(), itemPrefs, ctx);

    verify(ctx, itemPrefs);
  }

  public void testCosineSimilarityReducer() throws Exception {
    Reducer.Context ctx = createMock(Reducer.Context.class);

    ctx.write(new ItemItemWritable(12l, 34l), new DoubleWritable(0.5d));

    replay(ctx);

    new CosineSimilarityReducer().reduce(new ItemPairWritable(12l, 34l, 20d),
        Arrays.asList(new FloatWritable(5f), new FloatWritable(5f)), ctx);

    verify(ctx);
  }

  public void testCompleteJob() throws Exception {

    String tmpDirPath = System.getProperty("java.io.tmpdir")+"/"+ItemSimilarityTest.class.getCanonicalName();
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

      similarityJob.setConf(conf);

      similarityJob.run(new String[] { "--tempDir", tmpDirPath+"/tmp"});

      BufferedReader reader = new BufferedReader(new FileReader(tmpDirPath+"/output/part-r-00000"));

      String line = null;
      int currentLine = 1;
      while ( (line = reader.readLine()) != null) {

        String[] tokens = line.split("\t");

        long itemAID = Long.parseLong(tokens[0]);
        long itemBID = Long.parseLong(tokens[1]);
        double similarity = Double.parseDouble(tokens[2]);

        if (currentLine == 1) {
          assertEquals(1l, itemAID);
          assertEquals(3l, itemBID);
          assertEquals(0.45, similarity, 0.01);
        }

        if (currentLine == 2) {
          assertEquals(2l, itemAID);
          assertEquals(3l, itemBID);
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
