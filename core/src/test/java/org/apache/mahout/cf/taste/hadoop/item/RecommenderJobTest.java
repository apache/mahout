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
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.cf.taste.hadoop.EntityPrefWritable;
import org.apache.mahout.cf.taste.hadoop.MutableRecommendedItem;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.ToItemPrefsMapper;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.GenericRecommendedItem;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.MathHelper;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.CooccurrenceCountSimilarity;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.TanimotoCoefficientSimilarity;
import org.apache.mahout.math.map.OpenIntLongHashMap;
import org.easymock.IArgumentMatcher;
import org.easymock.EasyMock;
import org.junit.Test;

public class RecommenderJobTest extends TasteTestCase {

  /**
   * tests {@link ItemIDIndexMapper}
   */
  @Test
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
   */
  @Test
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
   */
  @Test
  public void testToItemPrefsMapper() throws Exception {
    Mapper<LongWritable,Text, VarLongWritable,VarLongWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(new VarLongWritable(12L), new EntityPrefWritable(34L, 1.0f));
    context.write(new VarLongWritable(56L), new EntityPrefWritable(78L, 2.0f));
    EasyMock.replay(context);

    ToItemPrefsMapper mapper = new ToItemPrefsMapper();
    mapper.map(new LongWritable(123L), new Text("12,34,1"), context);
    mapper.map(new LongWritable(456L), new Text("56,78,2"), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link ToItemPrefsMapper} using boolean data
   */
  @Test
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
   * tests {@link ToUserVectorsReducer}
   */
  @Test
  public void testToUserVectorReducer() throws Exception {
    Reducer<VarLongWritable,VarLongWritable,VarLongWritable,VectorWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);
    Counter userCounters = EasyMock.createMock(Counter.class);

    EasyMock.expect(context.getCounter(ToUserVectorsReducer.Counters.USERS)).andReturn(userCounters);
    userCounters.increment(1);
    context.write(EasyMock.eq(new VarLongWritable(12L)), MathHelper.vectorMatches(
        MathHelper.elem(TasteHadoopUtils.idToIndex(34L), 1.0), MathHelper.elem(TasteHadoopUtils.idToIndex(56L), 2.0)));

    EasyMock.replay(context, userCounters);

    Collection<VarLongWritable> varLongWritables = Lists.newLinkedList();
    varLongWritables.add(new EntityPrefWritable(34L, 1.0f));
    varLongWritables.add(new EntityPrefWritable(56L, 2.0f));

    new ToUserVectorsReducer().reduce(new VarLongWritable(12L), varLongWritables, context);

    EasyMock.verify(context, userCounters);
  }

  /**
   * tests {@link ToUserVectorsReducer} using boolean data
   */
  @Test
  public void testToUserVectorReducerWithBooleanData() throws Exception {
    Reducer<VarLongWritable,VarLongWritable,VarLongWritable,VectorWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);
    Counter userCounters = EasyMock.createMock(Counter.class);

    EasyMock.expect(context.getCounter(ToUserVectorsReducer.Counters.USERS)).andReturn(userCounters);
    userCounters.increment(1);
    context.write(EasyMock.eq(new VarLongWritable(12L)), MathHelper.vectorMatches(
        MathHelper.elem(TasteHadoopUtils.idToIndex(34L), 1.0), MathHelper.elem(TasteHadoopUtils.idToIndex(56L), 1.0)));

    EasyMock.replay(context, userCounters);

    new ToUserVectorsReducer().reduce(new VarLongWritable(12L), Arrays.asList(new VarLongWritable(34L),
        new VarLongWritable(56L)), context);

    EasyMock.verify(context, userCounters);
  }

  /**
   * tests {@link SimilarityMatrixRowWrapperMapper}
   */
  @Test
  public void testSimilarityMatrixRowWrapperMapper() throws Exception {
    Mapper<IntWritable,VectorWritable,VarIntWritable,VectorOrPrefWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(12)), vectorOfVectorOrPrefWritableMatches(MathHelper.elem(34, 0.5),
        MathHelper.elem(56, 0.7)));

    EasyMock.replay(context);

    RandomAccessSparseVector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    vector.set(12, 1.0);
    vector.set(34, 0.5);
    vector.set(56, 0.7);

    new SimilarityMatrixRowWrapperMapper().map(new IntWritable(12), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * verifies the {@link Vector} included in a {@link VectorOrPrefWritable}
   */
  private static VectorOrPrefWritable vectorOfVectorOrPrefWritableMatches(final Vector.Element... elements) {
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
   */
  @Test
  public void testUserVectorSplitterMapper() throws Exception {
    Mapper<VarLongWritable,VectorWritable, VarIntWritable,VectorOrPrefWritable>.Context context =
        EasyMock.createMock(Mapper.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(34)), prefOfVectorOrPrefWritableMatches(123L, 0.5f));
    context.write(EasyMock.eq(new VarIntWritable(56)), prefOfVectorOrPrefWritableMatches(123L, 0.7f));

    EasyMock.replay(context);

    UserVectorSplitterMapper mapper = new UserVectorSplitterMapper();
    setField(mapper, "maxPrefsPerUserConsidered", 10);

    RandomAccessSparseVector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    vector.set(34, 0.5);
    vector.set(56, 0.7);

    mapper.map(new VarLongWritable(123L), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * verifies a preference in a {@link VectorOrPrefWritable}
   */
  private static VectorOrPrefWritable prefOfVectorOrPrefWritableMatches(final long userID, final float prefValue) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorOrPrefWritable) {
          VectorOrPrefWritable pref = (VectorOrPrefWritable) argument;
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
   */
  @Test
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
    vector.set(34, 0.5);
    vector.set(56, 0.7);

    mapper.map(new VarLongWritable(123L), new VectorWritable(vector), context);
    mapper.map(new VarLongWritable(456L), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link UserVectorSplitterMapper} in the special case that the number of preferences to be considered
   * is less than the number of available preferences
   */
  @Test
  public void testUserVectorSplitterMapperOnlySomePrefsConsidered() throws Exception {
    Mapper<VarLongWritable,VectorWritable, VarIntWritable,VectorOrPrefWritable>.Context context =
        EasyMock.createMock(Mapper.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(34)), prefOfVectorOrPrefWritableMatchesNaN(123L));
    context.write(EasyMock.eq(new VarIntWritable(56)), prefOfVectorOrPrefWritableMatches(123L, 0.7f));

    EasyMock.replay(context);

    UserVectorSplitterMapper mapper = new UserVectorSplitterMapper();
    setField(mapper, "maxPrefsPerUserConsidered", 1);

    RandomAccessSparseVector vector = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    vector.set(34, 0.5);
    vector.set(56, 0.7);

    mapper.map(new VarLongWritable(123L), new VectorWritable(vector), context);

    EasyMock.verify(context);
  }

  /**
   * verifies that a preference value is NaN in a {@link VectorOrPrefWritable}
   */
  private static VectorOrPrefWritable prefOfVectorOrPrefWritableMatchesNaN(final long userID) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorOrPrefWritable) {
          VectorOrPrefWritable pref = (VectorOrPrefWritable) argument;
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
   */
  @Test
  public void testToVectorAndPrefReducer() throws Exception {
    Reducer<VarIntWritable,VectorOrPrefWritable,VarIntWritable,VectorAndPrefsWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarIntWritable(1)), vectorAndPrefsWritableMatches(Arrays.asList(123L, 456L),
        Arrays.asList(1.0f, 2.0f), MathHelper.elem(3, 0.5), MathHelper.elem(7, 0.8)));

    EasyMock.replay(context);

    Vector similarityColumn = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumn.set(3, 0.5);
    similarityColumn.set(7, 0.8);

    VectorOrPrefWritable itemPref1 = new VectorOrPrefWritable(123L, 1.0f);
    VectorOrPrefWritable itemPref2 = new VectorOrPrefWritable(456L, 2.0f);
    VectorOrPrefWritable similarities = new VectorOrPrefWritable(similarityColumn);

    new ToVectorAndPrefReducer().reduce(new VarIntWritable(1), Arrays.asList(itemPref1, itemPref2, similarities),
        context);

    EasyMock.verify(context);
  }

  /**
   * verifies a {@link VectorAndPrefsWritable}
   */
  private static VectorAndPrefsWritable vectorAndPrefsWritableMatches(final List<Long> userIDs,
      final List<Float> prefValues, final Vector.Element... elements) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorAndPrefsWritable) {
          VectorAndPrefsWritable vectorAndPrefs = (VectorAndPrefsWritable) argument;

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
   */
  @Test
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
    } catch (IllegalStateException e) {
      // good
    }

    EasyMock.verify(context);
  }

  /**
   * tests {@link org.apache.mahout.cf.taste.hadoop.item.ItemFilterMapper}
   */
  @Test
  public void testItemFilterMapper() throws Exception {

    Mapper<LongWritable,Text,VarLongWritable,VarLongWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    context.write(new VarLongWritable(34L), new VarLongWritable(12L));
    context.write(new VarLongWritable(78L), new VarLongWritable(56L));

    EasyMock.replay(context);

    ItemFilterMapper mapper = new ItemFilterMapper();
    mapper.map(null, new Text("12,34"), context);
    mapper.map(null, new Text("56,78"), context);

    EasyMock.verify(context);
  }

  /**
   * tests {@link org.apache.mahout.cf.taste.hadoop.item.ItemFilterAsVectorAndPrefsReducer}
   */
  @Test
  public void testItemFilterAsVectorAndPrefsReducer() throws Exception {
    Reducer<VarLongWritable,VarLongWritable,VarIntWritable,VectorAndPrefsWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);

    int itemIDIndex = TasteHadoopUtils.idToIndex(123L);
    context.write(EasyMock.eq(new VarIntWritable(itemIDIndex)), vectorAndPrefsForFilteringMatches(123L, 456L, 789L));

    EasyMock.replay(context);

    new ItemFilterAsVectorAndPrefsReducer().reduce(new VarLongWritable(123L), Arrays.asList(new VarLongWritable(456L),
        new VarLongWritable(789L)), context);

    EasyMock.verify(context);
  }

  static VectorAndPrefsWritable vectorAndPrefsForFilteringMatches(final long itemID, final long... userIDs) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof VectorAndPrefsWritable) {
          VectorAndPrefsWritable vectorAndPrefs = (VectorAndPrefsWritable) argument;
          Vector vector = vectorAndPrefs.getVector();
          if (vector.getNumNondefaultElements() != 1) {
            return false;
          }
          if (!Double.isNaN(vector.get(TasteHadoopUtils.idToIndex(itemID)))) {
            return false;
          }
          if (userIDs.length != vectorAndPrefs.getUserIDs().size()) {
            return false;
          }
          for (long userID : userIDs) {
            if (!vectorAndPrefs.getUserIDs().contains(userID)) {
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

  /**
   * tests {@link PartialMultiplyMapper}
   */
  @Test
  public void testPartialMultiplyMapper() throws Exception {

    Vector similarityColumn = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumn.set(3, 0.5);
    similarityColumn.set(7, 0.8);

    Mapper<VarIntWritable,VectorAndPrefsWritable,VarLongWritable,PrefAndSimilarityColumnWritable>.Context context =
      EasyMock.createMock(Mapper.Context.class);

    PrefAndSimilarityColumnWritable one = new PrefAndSimilarityColumnWritable();
    PrefAndSimilarityColumnWritable two = new PrefAndSimilarityColumnWritable();
    one.set(1.0f, similarityColumn);
    two.set(3.0f, similarityColumn);

    context.write(EasyMock.eq(new VarLongWritable(123L)), EasyMock.eq(one));
    context.write(EasyMock.eq(new VarLongWritable(456L)), EasyMock.eq(two));

    EasyMock.replay(context);

    VectorAndPrefsWritable vectorAndPrefs = new VectorAndPrefsWritable(similarityColumn, Arrays.asList(123L, 456L),
        Arrays.asList(1.0f, 3.0f));

    new PartialMultiplyMapper().map(new VarIntWritable(1), vectorAndPrefs, context);

    EasyMock.verify(context);
  }


  /**
   * tests {@link AggregateAndRecommendReducer}
   */
  @Test
  public void testAggregateAndRecommendReducer() throws Exception {
    Reducer<VarLongWritable,PrefAndSimilarityColumnWritable,VarLongWritable,RecommendedItemsWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(123L)), recommendationsMatch(new MutableRecommendedItem(1L, 2.8f),
        new MutableRecommendedItem(2L, 2.0f)));

    EasyMock.replay(context);

    RandomAccessSparseVector similarityColumnOne = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnOne.set(1, 0.1);
    similarityColumnOne.set(2, 0.5);

    RandomAccessSparseVector similarityColumnTwo = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnTwo.set(1, 0.9);
    similarityColumnTwo.set(2, 0.5);

    List<PrefAndSimilarityColumnWritable> values = Arrays.asList(
        new PrefAndSimilarityColumnWritable(1.0f, similarityColumnOne),
        new PrefAndSimilarityColumnWritable(3.0f, similarityColumnTwo));

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
   */
  @Test
  public void testAggregateAndRecommendReducerExcludeRecommendationsBasedOnOneItem() throws Exception {
    Reducer<VarLongWritable,PrefAndSimilarityColumnWritable,VarLongWritable,RecommendedItemsWritable>.Context context =
        EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(123L)), recommendationsMatch(new MutableRecommendedItem(1L, 2.8f)));

    EasyMock.replay(context);

    RandomAccessSparseVector similarityColumnOne = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnOne.set(1, 0.1);

    RandomAccessSparseVector similarityColumnTwo = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnTwo.set(1, 0.9);
    similarityColumnTwo.set(2, 0.5);

    List<PrefAndSimilarityColumnWritable> values = Arrays.asList(
        new PrefAndSimilarityColumnWritable(1.0f, similarityColumnOne),
        new PrefAndSimilarityColumnWritable(3.0f, similarityColumnTwo));

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
   */
  @Test
  public void testAggregateAndRecommendReducerLimitNumberOfRecommendations() throws Exception {
    Reducer<VarLongWritable,PrefAndSimilarityColumnWritable,VarLongWritable,RecommendedItemsWritable>.Context context =
      EasyMock.createMock(Reducer.Context.class);

    context.write(EasyMock.eq(new VarLongWritable(123L)), recommendationsMatch(new MutableRecommendedItem(1L, 2.8f)));

    EasyMock.replay(context);

    RandomAccessSparseVector similarityColumnOne = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnOne.set(1, 0.1);
    similarityColumnOne.set(2, 0.5);

    RandomAccessSparseVector similarityColumnTwo = new RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    similarityColumnTwo.set(1, 0.9);
    similarityColumnTwo.set(2, 0.5);

    List<PrefAndSimilarityColumnWritable> values = Arrays.asList(
        new PrefAndSimilarityColumnWritable(1.0f, similarityColumnOne),
        new PrefAndSimilarityColumnWritable(3.0f, similarityColumnTwo));

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
   */
  static RecommendedItemsWritable recommendationsMatch(final RecommendedItem... items) {
    EasyMock.reportMatcher(new IArgumentMatcher() {
      @Override
      public boolean matches(Object argument) {
        if (argument instanceof RecommendedItemsWritable) {
          RecommendedItemsWritable recommendedItemsWritable = (RecommendedItemsWritable) argument;
          List<RecommendedItem> expectedItems = Arrays.asList(items);
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
   *  Prediction(donkey, berries) = (0.66 * 3 + 0.25 * 5) / (0.66 + 0.25)                   ~ 3,5
   *
   * </pre>
   */
  @Test
  public void testCompleteJob() throws Exception {

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File similaritiesOutputDir = getTestTempDir("outputSimilarities");
    similaritiesOutputDir.delete();
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

    Configuration conf = getConfiguration();
    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    recommenderJob.setConf(conf);

    recommenderJob.run(new String[] { "--tempDir", tmpDir.getAbsolutePath(), "--similarityClassname",
       TanimotoCoefficientSimilarity.class.getName(), "--numRecommendations", "4",
        "--outputPathForSimilarityMatrix", similaritiesOutputDir.getAbsolutePath() });

    Map<Long,List<RecommendedItem>> recommendations = readRecommendations(new File(outputDir, "part-r-00000"));
    assertEquals(4, recommendations.size());

    for (Entry<Long,List<RecommendedItem>> entry : recommendations.entrySet()) {
      long userID = entry.getKey();
      List<RecommendedItem> items = entry.getValue();
      assertNotNull(items);
      RecommendedItem item1 = items.get(0);

      if (userID == 1L) {
        assertEquals(1, items.size());
        assertEquals(4L, item1.getItemID());
        assertEquals(4.3, item1.getValue(), 0.05);
      }
      if (userID == 2L) {
        assertEquals(1, items.size());
        assertEquals(2L, item1.getItemID());
        assertEquals(3.3, item1.getValue(), 0.05);
      }
      if (userID == 3L) {
        assertEquals(2, items.size());
        assertEquals(3L, item1.getItemID());
        assertEquals(4.1, item1.getValue(), 0.05);
        RecommendedItem item2 = items.get(1);
        assertEquals(1L, item2.getItemID());
        assertEquals(3.7, item2.getValue(), 0.05);
      }
      if (userID == 4L) {
        assertEquals(2, items.size());
        assertEquals(2L, item1.getItemID());
        assertEquals(4.0, item1.getValue(), 0.05);
        RecommendedItem item2 = items.get(1);
        assertEquals(3L, item2.getItemID());
        assertEquals(3.5, item2.getValue(), 0.05);
      }
    }

    Map<Pair<Long, Long>, Double> similarities = readSimilarities(new File(similaritiesOutputDir, "part-r-00000"));
    assertEquals(6, similarities.size());

    assertEquals(0.25, similarities.get(new Pair<Long, Long>(1L, 2L)), EPSILON);
    assertEquals(0.6666666666666666, similarities.get(new Pair<Long, Long>(1L, 3L)), EPSILON);
    assertEquals(0.5, similarities.get(new Pair<Long, Long>(1L, 4L)), EPSILON);
    assertEquals(0.3333333333333333, similarities.get(new Pair<Long, Long>(2L, 3L)), EPSILON);
    assertEquals(0.25, similarities.get(new Pair<Long, Long>(2L, 4L)), EPSILON);
    assertEquals(0.25, similarities.get(new Pair<Long, Long>(3L, 4L)), EPSILON);
  }

  /**
   * small integration test for boolean data
   */
  @Test
  public void testCompleteJobBoolean() throws Exception {

    File inputFile = getTestTempFile("prefs.txt");
    File outputDir = getTestTempDir("output");
    outputDir.delete();
    File tmpDir = getTestTempDir("tmp");
    File usersFile = getTestTempFile("users.txt");
    writeLines(usersFile, "3");

    writeLines(inputFile,
        "1,1",
        "1,2",
        "1,3",
        "2,1",
        "2,3",
        "2,4",
        "3,2",
        "3,4",
        "4,1",
        "4,4");

    RecommenderJob recommenderJob = new RecommenderJob();

    Configuration conf = getConfiguration();
    conf.set("mapred.input.dir", inputFile.getAbsolutePath());
    conf.set("mapred.output.dir", outputDir.getAbsolutePath());
    conf.setBoolean("mapred.output.compress", false);

    recommenderJob.setConf(conf);

    recommenderJob.run(new String[] { "--tempDir", tmpDir.getAbsolutePath(), "--similarityClassname",
        CooccurrenceCountSimilarity.class.getName(), "--booleanData", "true",
        "--usersFile", usersFile.getAbsolutePath() });

    Map<Long,List<RecommendedItem>> recommendations = readRecommendations(new File(outputDir, "part-r-00000"));

    List<RecommendedItem> recommendedToCow = recommendations.get(3L);
    assertEquals(2, recommendedToCow.size());

    RecommendedItem item1 = recommendedToCow.get(0);
    RecommendedItem item2 = recommendedToCow.get(1);

    assertEquals(1L, item1.getItemID());
    assertEquals(3L, item2.getItemID());

    /* predicted pref must be the sum of similarities:
    *    item1: coocc(burger, hotdog) + coocc(burger, icecream) = 3 
    *    item2: coocc(berries, hotdog) + coocc(berries, icecream) = 2 */
    assertEquals(3, item1.getValue(), 0.05);
    assertEquals(2, item2.getValue(), 0.05);
  }

  /**
   * check whether the explicit user/item filter works
   */
  @Test
  public void testCompleteJobWithFiltering() throws Exception {

     File inputFile = getTestTempFile("prefs.txt");
     File userFile = getTestTempFile("users.txt");
     File filterFile = getTestTempFile("filter.txt");
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

     /* only compute recommendations for the donkey */
     writeLines(userFile, "4");
     /* do not recommend the hotdog for the donkey */
     writeLines(filterFile, "4,2");

     RecommenderJob recommenderJob = new RecommenderJob();

     Configuration conf = getConfiguration();
     conf.set("mapred.input.dir", inputFile.getAbsolutePath());
     conf.set("mapred.output.dir", outputDir.getAbsolutePath());
     conf.setBoolean("mapred.output.compress", false);

     recommenderJob.setConf(conf);

     recommenderJob.run(new String[] { "--tempDir", tmpDir.getAbsolutePath(), "--similarityClassname",
        TanimotoCoefficientSimilarity.class.getName(), "--numRecommendations", "1",
        "--usersFile", userFile.getAbsolutePath(), "--filterFile", filterFile.getAbsolutePath() });

     Map<Long,List<RecommendedItem>> recommendations = readRecommendations(new File(outputDir, "part-r-00000"));

     assertEquals(1, recommendations.size());
     assertTrue(recommendations.containsKey(4L));
     assertEquals(1, recommendations.get(4L).size());

     /* berries should have been recommended to the donkey */
     RecommendedItem recommendedItem = recommendations.get(4L).get(0);
     assertEquals(3L, recommendedItem.getItemID());
     assertEquals(3.5, recommendedItem.getValue(), 0.05);
   }

  static Map<Pair<Long,Long>, Double> readSimilarities(File file) throws IOException {
    Map<Pair<Long,Long>, Double> similarities = Maps.newHashMap();
    for (String line : new FileLineIterable(file)) {
      String[] parts = line.split("\t");
      similarities.put(new Pair<Long,Long>(Long.parseLong(parts[0]), Long.parseLong(parts[1])),
          Double.parseDouble(parts[2]));
    }
    return similarities;
  }

  static Map<Long,List<RecommendedItem>> readRecommendations(File file) throws IOException {
    Map<Long,List<RecommendedItem>> recommendations = Maps.newHashMap();
    for (String line : new FileLineIterable(file)) {

      String[] keyValue = line.split("\t");
      long userID = Long.parseLong(keyValue[0]);
      String[] tokens = keyValue[1].replaceAll("\\[", "")
          .replaceAll("\\]", "").split(",");

      List<RecommendedItem> items = Lists.newLinkedList();
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
