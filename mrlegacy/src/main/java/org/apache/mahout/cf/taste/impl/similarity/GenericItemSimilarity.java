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

package org.apache.mahout.cf.taste.impl.similarity;

import java.util.Collection;
import java.util.Iterator;

import com.google.common.collect.AbstractIterator;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.common.RandomUtils;

import com.google.common.base.Preconditions;

/**
 * <p>
 * A "generic" {@link ItemSimilarity} which takes a static list of precomputed item similarities and bases its
 * responses on that alone. The values may have been precomputed offline by another process, stored in a file,
 * and then read and fed into an instance of this class.
 * </p>
 * 
 * <p>
 * This is perhaps the best {@link ItemSimilarity} to use with
 * {@link org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender}, for now, since the point
 * of item-based recommenders is that they can take advantage of the fact that item similarity is relatively
 * static, can be precomputed, and then used in computation to gain a significant performance advantage.
 * </p>
 */
public final class GenericItemSimilarity implements ItemSimilarity {

  private static final long[] NO_IDS = new long[0];
  
  private final FastByIDMap<FastByIDMap<Double>> similarityMaps = new FastByIDMap<FastByIDMap<Double>>();
  private final FastByIDMap<FastIDSet> similarItemIDsIndex = new FastByIDMap<FastIDSet>();

  /**
   * <p>
   * Creates a {@link GenericItemSimilarity} from a precomputed list of {@link ItemItemSimilarity}s. Each
   * represents the similarity between two distinct items. Since similarity is assumed to be symmetric, it is
   * not necessary to specify similarity between item1 and item2, and item2 and item1. Both are the same. It
   * is also not necessary to specify a similarity between any item and itself; these are assumed to be 1.0.
   * </p>
   *
   * <p>
   * Note that specifying a similarity between two items twice is not an error, but, the later value will win.
   * </p>
   *
   * @param similarities
   *          set of {@link ItemItemSimilarity}s on which to base this instance
   */
  public GenericItemSimilarity(Iterable<ItemItemSimilarity> similarities) {
    initSimilarityMaps(similarities.iterator());
  }

  /**
   * <p>
   * Like {@link #GenericItemSimilarity(Iterable)}, but will only keep the specified number of similarities
   * from the given {@link Iterable} of similarities. It will keep those with the highest similarity -- those
   * that are therefore most important.
   * </p>
   * 
   * <p>
   * Thanks to tsmorton for suggesting this and providing part of the implementation.
   * </p>
   * 
   * @param similarities
   *          set of {@link ItemItemSimilarity}s on which to base this instance
   * @param maxToKeep
   *          maximum number of similarities to keep
   */
  public GenericItemSimilarity(Iterable<ItemItemSimilarity> similarities, int maxToKeep) {
    Iterable<ItemItemSimilarity> keptSimilarities =
        TopItems.getTopItemItemSimilarities(maxToKeep, similarities.iterator());
    initSimilarityMaps(keptSimilarities.iterator());
  }

  /**
   * <p>
   * Builds a list of item-item similarities given an {@link ItemSimilarity} implementation and a
   * {@link DataModel}, rather than a list of {@link ItemItemSimilarity}s.
   * </p>
   * 
   * <p>
   * It's valid to build a {@link GenericItemSimilarity} this way, but perhaps missing some of the point of an
   * item-based recommender. Item-based recommenders use the assumption that item-item similarities are
   * relatively fixed, and might be known already independent of user preferences. Hence it is useful to
   * inject that information, using {@link #GenericItemSimilarity(Iterable)}.
   * </p>
   * 
   * @param otherSimilarity
   *          other {@link ItemSimilarity} to get similarities from
   * @param dataModel
   *          data model to get items from
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel} items
   */
  public GenericItemSimilarity(ItemSimilarity otherSimilarity, DataModel dataModel) throws TasteException {
    long[] itemIDs = GenericUserSimilarity.longIteratorToList(dataModel.getItemIDs());
    initSimilarityMaps(new DataModelSimilaritiesIterator(otherSimilarity, itemIDs));
  }

  /**
   * <p>
   * Like {@link #GenericItemSimilarity(ItemSimilarity, DataModel)} )}, but will only keep the specified
   * number of similarities from the given {@link DataModel}. It will keep those with the highest similarity
   * -- those that are therefore most important.
   * </p>
   * 
   * <p>
   * Thanks to tsmorton for suggesting this and providing part of the implementation.
   * </p>
   * 
   * @param otherSimilarity
   *          other {@link ItemSimilarity} to get similarities from
   * @param dataModel
   *          data model to get items from
   * @param maxToKeep
   *          maximum number of similarities to keep
   * @throws TasteException
   *           if an error occurs while accessing the {@link DataModel} items
   */
  public GenericItemSimilarity(ItemSimilarity otherSimilarity,
                               DataModel dataModel,
                               int maxToKeep) throws TasteException {
    long[] itemIDs = GenericUserSimilarity.longIteratorToList(dataModel.getItemIDs());
    Iterator<ItemItemSimilarity> it = new DataModelSimilaritiesIterator(otherSimilarity, itemIDs);
    Iterable<ItemItemSimilarity> keptSimilarities = TopItems.getTopItemItemSimilarities(maxToKeep, it);
    initSimilarityMaps(keptSimilarities.iterator());
  }

  private void initSimilarityMaps(Iterator<ItemItemSimilarity> similarities) {
    while (similarities.hasNext()) {
      ItemItemSimilarity iic = similarities.next();
      long similarityItemID1 = iic.getItemID1();
      long similarityItemID2 = iic.getItemID2();
      if (similarityItemID1 != similarityItemID2) {
        // Order them -- first key should be the "smaller" one
        long itemID1;
        long itemID2;
        if (similarityItemID1 < similarityItemID2) {
          itemID1 = similarityItemID1;
          itemID2 = similarityItemID2;
        } else {
          itemID1 = similarityItemID2;
          itemID2 = similarityItemID1;
        }
        FastByIDMap<Double> map = similarityMaps.get(itemID1);
        if (map == null) {
          map = new FastByIDMap<Double>();
          similarityMaps.put(itemID1, map);
        }
        map.put(itemID2, iic.getValue());

        doIndex(itemID1, itemID2);
        doIndex(itemID2, itemID1);
      }
      // else similarity between item and itself already assumed to be 1.0
    }
  }

  private void doIndex(long fromItemID, long toItemID) {
    FastIDSet similarItemIDs = similarItemIDsIndex.get(fromItemID);
    if (similarItemIDs == null) {
      similarItemIDs = new FastIDSet();
      similarItemIDsIndex.put(fromItemID, similarItemIDs);
    }
    similarItemIDs.add(toItemID);
  }

  /**
   * <p>
   * Returns the similarity between two items. Note that similarity is assumed to be symmetric, that
   * {@code itemSimilarity(item1, item2) == itemSimilarity(item2, item1)}, and that
   * {@code itemSimilarity(item1,item1) == 1.0} for all items.
   * </p>
   *
   * @param itemID1
   *          first item
   * @param itemID2
   *          second item
   * @return similarity between the two
   */
  @Override
  public double itemSimilarity(long itemID1, long itemID2) {
    if (itemID1 == itemID2) {
      return 1.0;
    }
    long firstID;
    long secondID;
    if (itemID1 < itemID2) {
      firstID = itemID1;
      secondID = itemID2;
    } else {
      firstID = itemID2;
      secondID = itemID1;
    }
    FastByIDMap<Double> nextMap = similarityMaps.get(firstID);
    if (nextMap == null) {
      return Double.NaN;
    }
    Double similarity = nextMap.get(secondID);
    return similarity == null ? Double.NaN : similarity;
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) {
    int length = itemID2s.length;
    double[] result = new double[length];
    for (int i = 0; i < length; i++) {
      result[i] = itemSimilarity(itemID1, itemID2s[i]);
    }
    return result;
  }

  @Override
  public long[] allSimilarItemIDs(long itemID) {
    FastIDSet similarItemIDs = similarItemIDsIndex.get(itemID);
    return similarItemIDs != null ? similarItemIDs.toArray() : NO_IDS;
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
  // Do nothing
  }
  
  /** Encapsulates a similarity between two items. Similarity must be in the range [-1.0,1.0]. */
  public static final class ItemItemSimilarity implements Comparable<ItemItemSimilarity> {
    
    private final long itemID1;
    private final long itemID2;
    private final double value;
    
    /**
     * @param itemID1
     *          first item
     * @param itemID2
     *          second item
     * @param value
     *          similarity between the two
     * @throws IllegalArgumentException
     *           if value is NaN, less than -1.0 or greater than 1.0
     */
    public ItemItemSimilarity(long itemID1, long itemID2, double value) {
      Preconditions.checkArgument(value >= -1.0 && value <= 1.0, "Illegal value: " + value + ". Must be: -1.0 <= value <= 1.0");
      this.itemID1 = itemID1;
      this.itemID2 = itemID2;
      this.value = value;
    }
    
    public long getItemID1() {
      return itemID1;
    }
    
    public long getItemID2() {
      return itemID2;
    }
    
    public double getValue() {
      return value;
    }
    
    @Override
    public String toString() {
      return "ItemItemSimilarity[" + itemID1 + ',' + itemID2 + ':' + value + ']';
    }
    
    /** Defines an ordering from highest similarity to lowest. */
    @Override
    public int compareTo(ItemItemSimilarity other) {
      double otherValue = other.getValue();
      return value > otherValue ? -1 : value < otherValue ? 1 : 0;
    }
    
    @Override
    public boolean equals(Object other) {
      if (!(other instanceof ItemItemSimilarity)) {
        return false;
      }
      ItemItemSimilarity otherSimilarity = (ItemItemSimilarity) other;
      return otherSimilarity.getItemID1() == itemID1
          && otherSimilarity.getItemID2() == itemID2
          && otherSimilarity.getValue() == value;
    }
    
    @Override
    public int hashCode() {
      return (int) itemID1 ^ (int) itemID2 ^ RandomUtils.hashDouble(value);
    }
    
  }
  
  private static final class DataModelSimilaritiesIterator extends AbstractIterator<ItemItemSimilarity> {
    
    private final ItemSimilarity otherSimilarity;
    private final long[] itemIDs;
    private int i;
    private long itemID1;
    private int j;

    private DataModelSimilaritiesIterator(ItemSimilarity otherSimilarity, long[] itemIDs) {
      this.otherSimilarity = otherSimilarity;
      this.itemIDs = itemIDs;
      i = 0;
      itemID1 = itemIDs[0];
      j = 1;
    }

    @Override
    protected ItemItemSimilarity computeNext() {
      int size = itemIDs.length;
      ItemItemSimilarity result = null;
      while (result == null && i < size - 1) {
        long itemID2 = itemIDs[j];
        double similarity;
        try {
          similarity = otherSimilarity.itemSimilarity(itemID1, itemID2);
        } catch (TasteException te) {
          // ugly:
          throw new IllegalStateException(te);
        }
        if (!Double.isNaN(similarity)) {
          result = new ItemItemSimilarity(itemID1, itemID2, similarity);
        }
        if (++j == size) {
          itemID1 = itemIDs[++i];
          j = i + 1;
        }
      }
      if (result == null) {
        return endOfData();
      } else {
        return result;
      }
    }
    
  }
  
}
