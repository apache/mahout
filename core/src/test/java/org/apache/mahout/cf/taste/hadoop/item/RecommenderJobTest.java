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

package org.apache.mahout.cf.taste.hadoop.item;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.FileLineIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.MathHelper;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedTanimotoCoefficientVectorSimilarity;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.easymock.IArgumentMatcher;
import org.easymock.classextension.EasyMock;

public class RecommenderJobTest extends TasteTestCase {

  /**
   * tests {@link ItemIDIndexMapper}
   *
   * @throws Exception
   */
  public void testItemIDIndexMapper() throws Exception {
    Mapper<LongWritable,Text, VarIntWritable, VarLongWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(new VarIntWritable(TasteHadoopUtils.idToIndex(789L)), new VarLongWritable(789L));
    EasyMock.replay(context);

    new ItemIDIndexMapper().map(new LongWritable(123L), new Text("456,789,5.0"), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link ItemIDIndexReducer}
   *
   * @throws Exception
   */
  public void testItemIDIndexReducer() throws Exception {
    Reducer<VarIntWritable, VarLongWritable, VarIntWritable,VarLongWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(new VarIntWritable(123), new VarLongWritable(45L));
    EasyMock.replay(context);

    new ItemIDIndexReducer().reduce(new VarIntWritable(123), Arrays.asList(new VarLongWritable(67L),
        new VarLongWritable(89L), new VarLongWritable(45L)), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link ToItemPrefsMapper}
   *
   * @throws Exception
   */
  public void testToItemPrefsMapper() throws Exception {
    Mapper<LongWritable,Text, VarLongWritable,VarLongWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(new VarLongWritable(12L), new EntityPrefWritable(34L, 1f));
    context.write(new VarLongWritable(56L), new EntityPrefWritable(78L, 2f));
    EasyMock.replay(context);

    ToItemPrefsMapper mapper = new ToItemPrefsMapper();
    mapper.map(new LongWritable(123L), new Text("12,34,1"), context);
    mapper.map(new LongWritable(456L), new Text("56,78,2"), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link ToItemPrefsMapper} using boolean data
   *
   * @throws Exception
   */
  public void testToItemPrefsMapperBooleanData() throws Exception {
    Mapper<LongWritable,Text, VarLongWritable,VarLongWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(new VarLongWritable(12L), new VarLongWritable(34L));
    context.write(new VarLongWritable(56L), new VarLongWritable(78L));
    EasyMock.replay(context);

    ToItemPrefsMapper mapper = new ToItemPrefsMapper();
    setField(mapper, "booleanData", true);
    mapper.map(new LongWritable(123L), new Text("12,34"), context);
    mapper.map(new LongWritable(456L), new Text("56,78"), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link ToUserVectorReducer}
   *
   * @throws Exception
   */
  public void testToUserVectorReducer() throws Exception {
    Reducer<VarLongWritable,VarLongWritable,VarLongWritable,VectorWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(12L)), MathHelper.vectorMatches(
        MathHelper.elem(TasteHadoopUtils.idToIndex(34L), 1d), MathHelper.elem(TasteHadoopUtils.idToIndex(56L), 2d)));

    EasyMock.replay(context);

    List<VarLongWritable> varLongWritables = new LinkedList<VarLongWritable>();
    varLongWritables.add(new EntityPrefWritable(34L, 1f));
    varLongWritables.add(new EntityPrefWritable(56L, 2f));

    new ToUserVectorReducer().reduce(new VarLongWritable(12L), varLongWritables, context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link ToUserVectorReducer} using boolean data
   *
   * @throws Exception
   */
  public void testToUserVectorReducerWithBooleanData() throws Exception {
    Reducer<VarLongWritable,VarLongWritable,VarLongWritable,VectorWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(12L)), MathHelper.vectorMatches(
        MathHelper.elem(TasteHadoopUtils.idToIndex(34L), 1d), MathHelper.elem(TasteHadoopUtils.idToIndex(56L), 1d)));

    EasyMock.replay(context);

    new ToUserVectorReducer().reduce(new VarLongWritable(12L), Arrays.asList(new VarLongWritable(34L),
        new VarLongWritable(56L)), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link SimilarityMatrixRowWrapperMapper}
   *
   * @throws Exception
   */
  public void testSimilarityMatrixRowWrapperMapper() throws Exception {
    Mapper<IntWritable,VectorWritable,VarIntWritable,VectorOrPrefWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(12)), vectorOfVectorOrPrefWritableMatches(MathHelper.elem(34, 0.5d),
        MathHelper.elem(56, 0.7d)));

    EasyMock.replay(context);

    RandomAccessSparseVector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    vector.set(12, 1d);
    vector.set(34, 0.5d);
    vector.set(56, 0.7d);

    new SimilarityMatrixRowWrapperMapper().map(new IntWritable(12), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * verifies the {@link Vector} included in a {@link VectorOrPrefWritable}
   *
   * @param elements
   * @return
   */
  public static VectorOrPrefWritable vectorOfVectorOrPrefWritableMatches(final Vector.Element... elements) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorOrPrefWritable) {
          Vector v = ((VectorOrPrefWritable) argument).getVector();
          return MathHelper.consistsOf(v, elements);
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
  }

  /**
   * tests {@link UserVectorSplitterMapper}
   *
   * @throws Exception
   */
  public void testUserVectorSplitterMapper() throws Exception {
    Mapper<VarLongWritable,VectorWritable, VarIntWritable,VectorOrPrefWritable>.Context context =
        EasyMock.createMock(Mapper.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(34)), prefOfVectorOrPrefWritableMatches(123L, 0.5f));
    context.write(EasyMock.eq(new VarIntWritable(56)), prefOfVectorOrPrefWritableMatches(123L, 0.7f));

    EasyMock.replay(context);

    UserVectorSplitterMapper mapper = new UserVectorSplitterMapper();
    setField(mapper, "maxPrefsPerUserConsidered", 10);

    RandomAccessSparseVector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    vector.set(34, 0.5d);
    vector.set(56, 0.7d);

    mapper.map(new VarLongWritable(123L), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * verifies a preference in a {@link VectorOrPrefWritable}
   *
   * @param userID
   * @param prefValue
   * @return
   */
  public static VectorOrPrefWritable prefOfVectorOrPrefWritableMatches(final long userID, final float prefValue) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorOrPrefWritable) {
          VectorOrPrefWritable pref = ((VectorOrPrefWritable) argument);
          return pref.getUserID() == userID && pref.getValue() == prefValue;
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
  }

  /**
   * tests {@link UserVectorSplitterMapper} in the special case that some userIDs shall be excluded
   *
   * @throws Exception
   */
  public void testUserVectorSplitterMapperUserExclusion() throws Exception {
    Mapper<VarLongWritable,VectorWritable, VarIntWritable,VectorOrPrefWritable>.Context context =
        EasyMock.createMock(Mapper.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(34)), prefOfVectorOrPrefWritableMatches(123L, 0.5f));
    context.write(EasyMock.eq(new VarIntWritable(56)), prefOfVectorOrPrefWritableMatches(123L, 0.7f));

    EasyMock.replay(context);

    FastIDSet usersToRecommendFor = new FastIDSet();
    usersToRecommendFor.add(123L);

    UserVectorSplitterMapper mapper = new UserVectorSplitterMapper();
    setField(mapper, "maxPrefsPerUserConsidered", 10);
    setField(mapper, "usersToRecommendFor", usersToRecommendFor);


    RandomAccessSparseVector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    vector.set(34, 0.5d);
    vector.set(56, 0.7d);

    mapper.map(new VarLongWritable(123L), new VectorWritable(vector), context);
    mapper.map(new VarLongWritable(456L), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link UserVectorSplitterMapper} in the special case that the number of preferences to be considered
   * is less than the number of available preferences
   *
   * @throws Exception
   */
  public void testUserVectorSplitterMapperOnlySomePrefsConsidered() throws Exception {
    Mapper<VarLongWritable,VectorWritable, VarIntWritable,VectorOrPrefWritable>.Context context =
        EasyMock.createMock(Mapper.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(34)), prefOfVectorOrPrefWritableMatchesNaN(123L));
    context.write(EasyMock.eq(new VarIntWritable(56)), prefOfVectorOrPrefWritableMatches(123L, 0.7f));

    EasyMock.replay(context);

    UserVectorSplitterMapper mapper = new UserVectorSplitterMapper();
    setField(mapper, "maxPrefsPerUserConsidered", 1);

    RandomAccessSparseVector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    vector.set(34, 0.5d);
    vector.set(56, 0.7d);

    mapper.map(new VarLongWritable(123L), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * verifies that a preference value is NaN in a {@link VectorOrPrefWritable}
   *
   * @param userID
   * @return
   */
  public static VectorOrPrefWritable prefOfVectorOrPrefWritableMatchesNaN(final long userID) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorOrPrefWritable) {
          VectorOrPrefWritable pref = ((VectorOrPrefWritable) argument);
          return pref.getUserID() == userID && Float.isNaN(pref.getValue());
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
  }

  /**
   * tests {@link ToVectorAndPrefReducer}
   *
   * @throws Exception
   */
  public void testToVectorAndPrefReducer() throws Exception {
    Reducer<VarIntWritable,VectorOrPrefWritable,VarIntWritable,VectorAndPrefsWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(1)), vectorAndPrefsWritableMatches(Arrays.asList(123L, 456L),
        Arrays.asList(1f, 2f), MathHelper.elem(3, 0.5d), MathHelper.elem(7, 0.8d)));

    EasyMock.replay(context);

    Vector similarityColumn = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumn.set(3, 0.5d);
    similarityColumn.set(7, 0.8d);

    VectorOrPrefWritable itemPref1 = new VectorOrPrefWritable(123L, 1f);
    VectorOrPrefWritable itemPref2 = new VectorOrPrefWritable(456L, 2f);
    VectorOrPrefWritable similarities = new VectorOrPrefWritable(similarityColumn);

    new ToVectorAndPrefReducer().reduce(new VarIntWritable(1), Arrays.asList(itemPref1, itemPref2, similarities),
        context);

    EasyMock.verify(context);
  }

  /**
   * verifies a {@link VectorAndPrefsWritable}
   *
   * @param userIDs
   * @param prefValues
   * @param elements
   * @return
   */
  public static VectorAndPrefsWritable vectorAndPrefsWritableMatches(final List<Long> userIDs,
      final List<Float> prefValues, final Vector.Element... elements) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorAndPrefsWritable) {
          VectorAndPrefsWritable vectorAndPrefs = ((VectorAndPrefsWritable) argument);

          if (!vectorAndPrefs.getUserIDs().equals(userIDs)) {
            return false;
          }
          if (!vectorAndPrefs.getValues().equals(prefValues)) {
            return false;
          }
          return MathHelper.consistsOf(vectorAndPrefs.getVector(), elements);
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
  }

  /**
   * tests {@link ToVectorAndPrefReducer} in the error case that two similarity column vectors a supplied for the same
   * item (which should never happen)
   *
   * @throws Exception
   */
  public void testToVectorAndPrefReducerExceptionOn2Vectors() throws Exception {
    Reducer<VarIntWritable,VectorOrPrefWritable,VarIntWritable,VectorAndPrefsWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    EasyMock.replay(context);

    Vector similarityColumn1 = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    Vector similarityColumn2 = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);

    VectorOrPrefWritable similarities1 = new VectorOrPrefWritable(similarityColumn1);
    VectorOrPrefWritable similarities2 = new VectorOrPrefWritable(similarityColumn2);

    try {
      new ToVectorAndPrefReducer().reduce(new VarIntWritable(1), Arrays.asList(similarities1, similarities2), context);
      fail();
    } catch (IllegalStateException e) {}

    EasyMock.verify(context);
  }

  /**
   * tests {@link PartialMultiplyMapper}
   *
   * @throws Exception
   */
  public void testPartialMultiplyMapper() throws Exception {

    Vector similarityColumn = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumn.set(3, 0.5d);
    similarityColumn.set(7, 0.8d);

    Mapper<VarIntWritable,VectorAndPrefsWritable,VarLongWritable,PrefAndSimilarityColumnWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    PrefAndSimilarityColumnWritable one = new PrefAndSimilarityColumnWritable();
    PrefAndSimilarityColumnWritable two = new PrefAndSimilarityColumnWritable();
    one.set(1f, similarityColumn);
    two.set(3f, similarityColumn);

    context.write(EasyMock.eq(new VarLongWritable(123L)), EasyMock.eq(one));
    context.write(EasyMock.eq(new VarLongWritable(456L)), EasyMock.eq(two));

    EasyMock.replay(context);

    VectorAndPrefsWritable vectorAndPrefs = new VectorAndPrefsWritable(similarityColumn, Arrays.asList(123L, 456L),
        Arrays.asList(1f, 3f));

    new PartialMultiplyMapper().map(new VarIntWritable(1), vectorAndPrefs, context);

    EasyMock.verify(context);
  }


  /**
   * tests {@link AggregateAndRecommendReducer}
   *
   * @throws Exception
   */
  public void testAggregateAndRecommendReducer() throws Exception {
    Reducer<VarLongWritable,PrefAndSimilarityColumnWritable,VarLongWritable,RecommendedItemsWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(123L)), recommendationsMatch(new GenericRecommendedItem(1L, 2.8f),
        new GenericRecommendedItem(2L, 2f)));

    EasyMock.replay(context);

    RandomAccessSparseVector similarityColumnOne = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnOne.set(1, 0.1d);
    similarityColumnOne.set(2, 0.5d);

    RandomAccessSparseVector similarityColumnTwo = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnTwo.set(1, 0.9d);
    similarityColumnTwo.set(2, 0.5d);

    List<PrefAndSimilarityColumnWritable> values = Arrays.asList(
        new PrefAndSimilarityColumnWritable(1f, similarityColumnOne),
        new PrefAndSimilarityColumnWritable(3f, similarityColumnTwo));

    OpenIntLongHashMap indexItemIDMap = new OpenIntLongHashMap();
    indexItemIDMap.put(1, 1L);
    indexItemIDMap.put(2, 2L);

    AggregateAndRecommendReducer reducer = new AggregateAndRecommendReducer();

    setField(reducer, "indexItemIDMap", indexItemIDMap);
    setField(reducer, "recommendationsPerUser", 3);

    reducer.reduce(new VarLongWritable(123L), values, context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link AggregateAndRecommendReducer}
   *
   * @throws Exception
   */
  public void testAggregateAndRecommendReducerExcludeRecommendationsBasedOnOneItem() throws Exception {
    Reducer<VarLongWritable,PrefAndSimilarityColumnWritable,VarLongWritable,RecommendedItemsWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(123L)), recommendationsMatch(new GenericRecommendedItem(1L, 2.8f)));

    EasyMock.replay(context);

    RandomAccessSparseVector similarityColumnOne = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnOne.set(1, 0.1d);

    RandomAccessSparseVector similarityColumnTwo = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnTwo.set(1, 0.9d);
    similarityColumnTwo.set(2, 0.5d);

    List<PrefAndSimilarityColumnWritable> values = Arrays.asList(
        new PrefAndSimilarityColumnWritable(1f, similarityColumnOne),
        new PrefAndSimilarityColumnWritable(3f, similarityColumnTwo));

    OpenIntLongHashMap indexItemIDMap = new OpenIntLongHashMap();
    indexItemIDMap.put(1, 1L);
    indexItemIDMap.put(2, 2L);

    AggregateAndRecommendReducer reducer = new AggregateAndRecommendReducer();

    setField(reducer, "indexItemIDMap", indexItemIDMap);
    setField(reducer, "recommendationsPerUser", 3);

    reducer.reduce(new VarLongWritable(123L), values, context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link AggregateAndRecommendReducer} with a limit on the recommendations per user
   *
   * @throws Exception
   */
  public void testAggregateAndRecommendReducerLimitNumberOfRecommendations() throws Exception {
    Reducer<VarLongWritable,PrefAndSimilarityColumnWritable,VarLongWritable,RecommendedItemsWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(123L)), recommendationsMatch(new GenericRecommendedItem(1L, 2.8f)));

    EasyMock.replay(context);

    RandomAccessSparseVector similarityColumnOne = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnOne.set(1, 0.1d);
    similarityColumnOne.set(2, 0.5d);

    RandomAccessSparseVector similarityColumnTwo = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnTwo.set(1, 0.9d);
    similarityColumnTwo.set(2, 0.5d);

    List<PrefAndSimilarityColumnWritable> values = Arrays.asList(
        new PrefAndSimilarityColumnWritable(1f, similarityColumnOne),
        new PrefAndSimilarityColumnWritable(3f, similarityColumnTwo));

    OpenIntLongHashMap indexItemIDMap = new OpenIntLongHashMap();
    indexItemIDMap.put(1, 1L);
    indexItemIDMap.put(2, 2L);

    AggregateAndRecommendReducer reducer = new AggregateAndRecommendReducer();

    setField(reducer, "indexItemIDMap", indexItemIDMap);
    setField(reducer, "recommendationsPerUser", 1);

    reducer.reduce(new VarLongWritable(123L), values, context);

    EasyMock.verify(context);
  }

  /**
   * verifies a {@link RecommendedItemsWritable}
   *
   * @param items
   * @return
   */
  static RecommendedItemsWritable recommendationsMatch(final RecommendedItem... items) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof RecommendedItemsWritable) {
          RecommendedItemsWritable recommendedItemsWritable = ((RecommendedItemsWritable) argument);
          List<RecommendedItem> expectedItems = new LinkedList<RecommendedItem>();
          for (RecommendedItem item : items) {
            expectedItems.add(item);
          }
          return expectedItems.equals(recommendedItemsWritable.getRecommendedItems());
        }
        return false;
      }

      @Override
      public void appendTo(StringBuffer buffer) {}
    });
    return null;
  }

  /**
   * small integration test that runs the full job
   *
   * As a tribute to http://www.slideshare.net/srowen/collaborative-filtering-at-scale,
   * we recommend people food to animals in this test :)
   *
   * <pre>
   *
   *  user-item-matrix
   *
   *          burger  hotdog  berries  icecream
   *  dog       5       5        2        -
   *  rabbit    2       -        3        5
   *  cow       -       5        -        3
   *  donkey    3       -        -        5
   *
   *
   *  item-item-similarity-matrix (tanimoto-coefficient of the item-vectors of the user-item-matrix)
   *
   *          burger  hotdog  berries icecream
   *  burger    -      0.25    0.66    0.5
   *  hotdog   0.25     -      0.33    0.25
   *  berries  0.66    0.33     -      0.25
   *  icecream 0.5     0.25    0.25     -
   *
   *
   *  Prediction(dog, icecream)   = (0.5 * 5 + 0.25 * 5 + 0.25 * 2 ) / (0.5 + 0.25 + 0.25)  ~ 4.3
   *  Prediction(rabbit, hotdog)  = (0.25 * 2 + 0.33 * 3 + 0.25 * 5) / (0.25 + 0.33 + 0.25) ~ 3,3
   *  Prediction(cow, burger)     = (0.25 * 5 + 0.5 * 3) / (0.25 + 0.5)                     ~ 3,7
   *  Prediction(cow, berries)    = (0.33 * 5 + 0.25 * 3) / (0.33 + 0.25)                   ~ 4,1
   *  Prediction(donkey, hotdog)  = (0.25 * 3 + 0.25 * 5) / (0.25 + 0.25)                   ~ 4
   *  Prediction(donkey, berries) = (0.66 * 3 + 0.25 * 5) / (0.66 + 0.25)                   ~ 3,6
   *
   * </pre>
   *
   *
   * @throws Exception
   */
  public void testCompleteJob() throws Exception {

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");

    writeLines(inputFile,
        "1,1,5",
        "1,2,5",
        "1,3,2",
        "2,1,2",
        "2,3,3",
        "2,4,5",
        "3,2,5",
        "3,4,3",
        "4,1,3",
        "4,4,5");

    RecommenderJob recommenderJob = new RecommenderJob();

    Configuration conf = new Configuration();
    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    recommenderJob.setConf(conf);

    recommenderJob.run(new String[] { "--tempDir", tmpDir.getAbsolutePath(), "--similarityClassname",
       DistributedTanimotoCoefficientVectorSimilarity.class.getName(), "--numRecommendations", String.valueOf(1) });

    Map<Long,List<RecommendedItem>> recommendations = readRecommendations(new File(outputDir, "part-r-00000"));

    assertEquals(4, recommendations.size());

    for (Entry<Long,List<RecommendedItem>> entry : recommendations.entrySet()) {
      long userID = entry.getKey();
      List<RecommendedItem> items = entry.getValue();
      assertNotNull(items);
      assertEquals(1, items.size());
      RecommendedItem item = items.get(0);

      if (userID == 1L) {
        assertEquals(4L, item.getItemID());
        assertEquals(4.3d, item.getValue(), 0.05d);
      }
      if (userID == 2L) {
        assertEquals(2L, item.getItemID());
        assertEquals(3.3d, item.getValue(), 0.05d);
      }
      if (userID == 3L) {
        assertEquals(3L, item.getItemID());
        assertEquals(4.1d, item.getValue(), 0.05d);
      }
      if (userID == 4L) {
        assertEquals(2L, item.getItemID());
        assertEquals(4d, item.getValue(), 0.05d);
      }
    }
  }

  static Map<Long,List<RecommendedItem>> readRecommendations(File file) throws IOException {
    Map<Long,List<RecommendedItem>> recommendations = new HashMap<Long,List<RecommendedItem>>();
    FileLineIterable lineIterable = new FileLineIterable(file);
    for (String line : lineIterable) {

      String[] keyValue = line.split("\t");
      long userID = Long.parseLong(keyValue[0]);
      String[] tokens = keyValue[1].replaceAll("\\[", "")
          .replaceAll("\\]", "").split(",");

      List<RecommendedItem> items = new LinkedList<RecommendedItem>();
      for (String token : tokens) {
        String[] itemTokens = token.split(":");
        long itemID = Long.parseLong(itemTokens[0]);
        float value = Float.parseFloat(itemTokens[1]);
        items.add(new GenericRecommendedItem(itemID, value));
      }
      recommendations.put(userID, items);
    }
    return recommendations;
  }

}
