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
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityWritable;
import org.apache.mahout.cf.taste.hadoop.ItemItemWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPairWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthArrayWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.ItemPrefWithLengthWritable;
import org.apache.mahout.cf.taste.hadoop.similarity.item.writables.UserPrefArrayWritable;
import org.apache.mahout.common.MahoutTestCase;
import org.easymock.IArgumentMatcher;
import org.easymock.classextension.EasyMock;

/**
 * Unit tests for the mappers and reducers in org.apache.mahout.cf.taste.hadoop.similarity
 * Integration test with a mini-file at the end
 *
 */
public class ItemSimilarityTest extends MahoutTestCase {


  public void testUserPrefsPerItemMapper() throws Exception {
    Mapper.Context ctx = createMock(Mapper.Context.class);
    ctx.write(new EntityWritable(34L), new EntityPrefWritable(12L, 2.3f));
    replay(ctx);

    new UserPrefsPerItemMapper().map(new LongWritable(), new Text("12,34,2.3"), ctx);

    verify(ctx);
  }

  public void testToItemVectorReducer() throws Exception {

    List<EntityPrefWritable> userPrefs = Arrays.asList(new EntityPrefWritable(34L, 1.0f), new EntityPrefWritable(56L, 2.0f));

    Reducer.Context ctx = createMock(Reducer.Context.class);

    ctx.write(eq(new EntityWritable(12L)), equalToUserPrefs(userPrefs));

    replay(ctx);

    new ToItemVectorReducer().reduce(new EntityWritable(12L), userPrefs, ctx);

    verify(ctx);
  }

  static UserPrefArrayWritable equalToUserPrefs(final Collection<EntityPrefWritable> prefsToCheck) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof UserPrefArrayWritable) {
          UserPrefArrayWritable userPrefArray = (UserPrefArrayWritable) argument;
          Set<EntityPrefWritable> set = new HashSet<EntityPrefWritable>();
          set.addAll(Arrays.asList(userPrefArray.getUserPrefs()));

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
    Mapper.Context ctx = createMock(Mapper.Context.class);
    UserPrefArrayWritable userPrefs = createMock(UserPrefArrayWritable.class);

    expect(userPrefs.getUserPrefs())
        .andReturn(new EntityPrefWritable[] { new EntityPrefWritable(12L, 2.0f), new EntityPrefWritable(56L, 3.0f) });

    double length = Math.sqrt(Math.pow(2.0f, 2) + Math.pow(3.0f, 2));

    ctx.write(new EntityWritable(12L), new ItemPrefWithLengthWritable(34L, length, 2.0f));
    ctx.write(new EntityWritable(56L), new ItemPrefWithLengthWritable(34L, length, 3.0f));

    replay(ctx, userPrefs);

    new PreferredItemsPerUserMapper().map(new EntityWritable(34L), userPrefs, ctx);

    verify(ctx, userPrefs);
  }

  public void testPreferredItemsPerUserReducer() throws Exception {

    List<ItemPrefWithLengthWritable> itemPrefs =
        Arrays.asList(new ItemPrefWithLengthWritable(34L, 5.0, 1.0f), new ItemPrefWithLengthWritable(56L, 7.0, 2.0f));

    Reducer.Context ctx = createMock(Reducer.Context.class);

    ctx.write(eq(new EntityWritable(12L)), equalToItemPrefs(itemPrefs));

    replay(ctx);

    new PreferredItemsPerUserReducer().reduce(new EntityWritable(12L), itemPrefs, ctx);

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
        new ItemPrefWithLengthWritable(34L, 2.0, 1.0f), new ItemPrefWithLengthWritable(56L, 3.0, 2.0f),
        new ItemPrefWithLengthWritable(78L, 4.0, 3.0f) });

    ctx.write(new ItemPairWritable(34L, 56L, 6.0), new FloatWritable(2.0f));
    ctx.write(new ItemPairWritable(34L, 78L, 8.0), new FloatWritable(3.0f));
    ctx.write(new ItemPairWritable(56L, 78L, 12.0), new FloatWritable(6.0f));

    replay(ctx, itemPrefs);

    new CopreferredItemsMapper().map(new EntityWritable(), itemPrefs, ctx);

    verify(ctx, itemPrefs);
  }

  public void testCosineSimilarityReducer() throws Exception {
    Reducer.Context ctx = createMock(Reducer.Context.class);

    ctx.write(new ItemItemWritable(12L, 34L), new DoubleWritable(0.5d));

    replay(ctx);

    new CosineSimilarityReducer().reduce(new ItemPairWritable(12L, 34L, 20.0),
        Arrays.asList(new FloatWritable(5.0f), new FloatWritable(5.0f)), ctx);

    verify(ctx);
  }

  public void testCompleteJob() throws Exception {

    String tmpDirPath = System.getProperty("java.io.tmpdir")+ '/' +ItemSimilarityTest.class.getCanonicalName();
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
