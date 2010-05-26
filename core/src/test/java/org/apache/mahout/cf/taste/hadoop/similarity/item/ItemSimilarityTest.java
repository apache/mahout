/*
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
import java.io.FilenameFilter;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritableArrayWritable;
import org.apache.mahout.cf.taste.hadoop.ToUserPrefsMapper;
import org.apache.mahout.cf.taste.hadoop.similarity.CoRating;
import org.apache.mahout.cf.taste.hadoop.similarity.DistributedUncenteredZeroAssumingCosineSimilarity;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.easymock.IArgumentMatcher;
import org.easymock.classextension.EasyMock;

/**
 * Unit tests for the mappers and reducers in org.apache.mahout.cf.taste.hadoop.similarity
 * Integration test with a mini-file at the end
 */
public final class ItemSimilarityTest extends MahoutTestCase {

  public void testUserPrefsPerItemMapper() throws Exception {
    Mapper<LongWritable,Text,VarLongWritable,VarLongWritable>.Context context =
        EasyMock.createMock(Mapper.Context.class);
    context.write(new VarLongWritable(34L), new EntityPrefWritable(12L, 2.3f));
    EasyMock.replay(context);

    new ToUserPrefsMapper().map(new LongWritable(), new Text("12,34,2.3"), context);

    EasyMock.verify(context);
  }

  public void testCountUsersMapper() throws Exception {
    Mapper<LongWritable,Text,CountUsersKeyWritable,VarLongWritable>.Context context =
        EasyMock.createMock(Mapper.Context.class);
    context.write(keyForUserID(12L), EasyMock.eq(new VarLongWritable(12L)));
    context.write(keyForUserID(35L), EasyMock.eq(new VarLongWritable(35L)));
    EasyMock.replay(context);

    CountUsersMapper mapper = new CountUsersMapper();
    mapper.map(null, new Text("12,100,1.3"), context);
    mapper.map(null, new Text("35,100,3.0"), context);

    EasyMock.verify(context);
  }

  static CountUsersKeyWritable keyForUserID(final long userID) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof CountUsersKeyWritable) {
          CountUsersKeyWritable key = (CountUsersKeyWritable) argument;
          return (userID == key.getUserID());
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });

    return null;
  }

  public void testCountUsersReducer() throws Exception {

    Reducer<CountUsersKeyWritable,VarLongWritable,VarIntWritable,NullWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);
    context.write(new VarIntWritable(3), NullWritable.get());
    EasyMock.replay(context);

    List<VarLongWritable> userIDs = Arrays.asList(new VarLongWritable(1L), new VarLongWritable(1L),
                                                  new VarLongWritable(3L), new VarLongWritable(5L),
                                                  new VarLongWritable(5L), new VarLongWritable(5L));

    new CountUsersReducer().reduce(null, userIDs, context);

    EasyMock.verify(context);
  }

  public void testToItemVectorReducer() throws Exception {

    List<EntityPrefWritable> userPrefs = Arrays.asList(
        new EntityPrefWritable(34L, 1.0f), new EntityPrefWritable(56L, 2.0f));

    Reducer<VarLongWritable,EntityPrefWritable,VarLongWritable,EntityPrefWritableArrayWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(12L)), equalToUserPrefs(userPrefs));

    EasyMock.replay(context);

    new ToItemVectorReducer().reduce(new VarLongWritable(12L), userPrefs, context);

    EasyMock.verify(context);
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
    Mapper<VarLongWritable,EntityPrefWritableArrayWritable,VarLongWritable,ItemPrefWithItemVectorWeightWritable>.Context context =
        EasyMock.createMock(Mapper.Context.class);
    EntityPrefWritableArrayWritable userPrefs = new EntityPrefWritableArrayWritable(
        new EntityPrefWritable[] {
            new EntityPrefWritable(12L, 2.0f),
            new EntityPrefWritable(56L, 3.0f) });

    Configuration conf = new Configuration();
    EasyMock.expect(context.getConfiguration()).andStubReturn(conf);

    double weight =
      new DistributedUncenteredZeroAssumingCosineSimilarity().weightOfItemVector(Arrays.asList(2.0f, 3.0f));

    context.write(new VarLongWritable(12L), new ItemPrefWithItemVectorWeightWritable(34L, weight, 2.0f));
    context.write(new VarLongWritable(56L), new ItemPrefWithItemVectorWeightWritable(34L, weight, 3.0f));

    conf.set(ItemSimilarityJob.DISTRIBUTED_SIMILARITY_CLASSNAME,
        "org.apache.mahout.cf.taste.hadoop.similarity.DistributedUncenteredZeroAssumingCosineSimilarity");

    EasyMock.replay(context);

    PreferredItemsPerUserMapper mapper = new PreferredItemsPerUserMapper();
    mapper.setup(context);
    mapper.map(new VarLongWritable(34L), userPrefs, context);

    EasyMock.verify(context);
  }

  public void testPreferredItemsPerUserReducer() throws Exception {

    List<ItemPrefWithItemVectorWeightWritable> itemPrefs =
        Arrays.asList(new ItemPrefWithItemVectorWeightWritable(34L, 5.0, 1.0f),
                      new ItemPrefWithItemVectorWeightWritable(56L, 7.0, 2.0f));

    Reducer<VarLongWritable,ItemPrefWithItemVectorWeightWritable,VarLongWritable,ItemPrefWithItemVectorWeightArrayWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(12L)), equalToItemPrefs(itemPrefs));

    EasyMock.replay(context);

    new PreferredItemsPerUserReducer().reduce(new VarLongWritable(12L), itemPrefs, context);

    EasyMock.verify(context);
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
    Mapper<VarLongWritable,ItemPrefWithItemVectorWeightArrayWritable,ItemPairWritable,CoRating>.Context context =
        EasyMock.createMock(Mapper.Context.class);
    ItemPrefWithItemVectorWeightArrayWritable itemPrefs =
        EasyMock.createMock(ItemPrefWithItemVectorWeightArrayWritable.class);

    EasyMock.expect(itemPrefs.getItemPrefs()).andReturn(new ItemPrefWithItemVectorWeightWritable[] {
        new ItemPrefWithItemVectorWeightWritable(34L, 2.0, 1.0f), new ItemPrefWithItemVectorWeightWritable(56L, 3.0, 2.0f),
        new ItemPrefWithItemVectorWeightWritable(78L, 4.0, 3.0f) });

    context.write(new ItemPairWritable(34L, 56L, 2.0, 3.0), new CoRating(1.0f, 2.0f));
    context.write(new ItemPairWritable(34L, 78L, 2.0, 4.0), new CoRating(1.0f, 3.0f));
    context.write(new ItemPairWritable(56L, 78L, 3.0, 4.0), new CoRating(2.0f, 3.0f));

    EasyMock.replay(context, itemPrefs);

    new CopreferredItemsMapper().map(new VarLongWritable(), itemPrefs, context);

    EasyMock.verify(context, itemPrefs);
  }

  public void testSimilarityReducer() throws Exception {
    Reducer<ItemPairWritable,CoRating,EntityEntityWritable,DoubleWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);
    Configuration conf = new Configuration();
    EasyMock.expect(context.getConfiguration()).andStubReturn(conf);

    conf.set(ItemSimilarityJob.DISTRIBUTED_SIMILARITY_CLASSNAME,
        "org.apache.mahout.cf.taste.hadoop.similarity.DistributedUncenteredZeroAssumingCosineSimilarity");
    conf.setInt(ItemSimilarityJob.NUMBER_OF_USERS, 1);

    context.write(new EntityEntityWritable(12L, 34L), new DoubleWritable(0.5));

    EasyMock.replay(context);

    SimilarityReducer reducer = new SimilarityReducer();
    reducer.setup(context);
    reducer.reduce(new ItemPairWritable(12L, 34L, 2.0, 10.0),
                   Arrays.asList(new CoRating(2.5f, 2.0f),new CoRating(2.0f, 2.5f)), context);

    EasyMock.verify(context);
  }

  public void testCompleteJob() throws Exception {

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    /* user-item-matrix

                 Game   Mouse   PC    Disk
         Jane     -       1      2      -
         Paul     1       -      1      -
         Fred     -       -      -      1
     */

    BufferedWriter writer = new BufferedWriter(new FileWriter(inputFile));
    try {
      writer.write("2,1,1\n" +
                   "1,2,1\n" +
                   "3,4,1\n" +
                   "1,3,2\n" +
                   "2,3,1\n");
    } finally {
      writer.close();
    }

    ItemSimilarityJob similarityJob = new ItemSimilarityJob();

    Configuration conf = new Configuration();
    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    similarityJob.setConf(conf);

    similarityJob.run(new String[] { "--tempDir", tmpDir.getAbsolutePath(), "--similarityClassname",
        "org.apache.mahout.cf.taste.hadoop.similarity.DistributedUncenteredZeroAssumingCosineSimilarity"});

    File countUsersPart = new File(tmpDir, "countUsers");
    int numberOfUsers = ItemSimilarityJob.readNumberOfUsers(new Configuration(),
                                                            new Path(countUsersPart.getAbsolutePath()));

    assertEquals(3, numberOfUsers);

    File outPart = outputDir.listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File dir, String name) {
        return name.startsWith("part-");
      }
    })[0];
    BufferedReader reader = new BufferedReader(new FileReader(outPart));

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

  }

}
