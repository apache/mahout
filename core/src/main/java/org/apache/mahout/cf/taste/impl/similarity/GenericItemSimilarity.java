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

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.IteratorIterable;
import org.apache.mahout.cf.taste.impl.common.IteratorUtils;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * <p>A "generic" {@link ItemSimilarity} which takes a static list of precomputed item similarities and bases
 * its responses on that alone. The values may have been precomputed offline by another process, stored in a file, and
 * then read and fed into an instance of this class.</p>
 *
 * <p>This is perhaps the best {@link ItemSimilarity} to use with {@link org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender},
 * for now, since the point of item-based recommenders is that they can take advantage of the fact that item similarity
 * is relatively static, can be precomputed, and then used in computation to gain a significant performance
 * advantage.</p>
 */
public final class GenericItemSimilarity implements ItemSimilarity {

  private final Map<Comparable<?>, Map<Comparable<?>, Double>> similarityMaps =
          new FastMap<Comparable<?>, Map<Comparable<?>, Double>>();

  /**
   * <p>Creates a {@link GenericItemSimilarity} from a precomputed list of {@link ItemItemSimilarity}s. Each represents
   * the similarity between two distinct items. Since similarity is assumed to be symmetric, it is not necessary to
   * specify similarity between item1 and item2, and item2 and item1. Both are the same. It is also not necessary to
   * specify a similarity between any item and itself; these are assumed to be 1.0.</p>
   *
   * <p>Note that specifying a similarity between two items twice is not an error, but, the later value will win.</p>
   *
   * @param similarities set of {@link ItemItemSimilarity}s on which to base this instance
   */
  public GenericItemSimilarity(Iterable<ItemItemSimilarity> similarities) {
    initSimilarityMaps(similarities);
  }

  /**
   * <p>Like {@link #GenericItemSimilarity(Iterable)}, but will only keep the specified number of similarities from the
   * given {@link Iterable} of similarities. It will keep those with the highest similarity -- those that are therefore
   * most important.</p>
   *
   * <p>Thanks to tsmorton for suggesting this and providing part of the implementation.</p>
   *
   * @param similarities set of {@link ItemItemSimilarity}s on which to base this instance
   * @param maxToKeep    maximum number of similarities to keep
   */
  public GenericItemSimilarity(Iterable<ItemItemSimilarity> similarities, int maxToKeep) {
    Iterable<ItemItemSimilarity> keptSimilarities = TopItems.getTopItemItemSimilarities(maxToKeep, similarities);
    initSimilarityMaps(keptSimilarities);
  }

  /**
   * <p>Builds a list of item-item similarities given an {@link ItemSimilarity} implementation and a {@link DataModel},
   * rather than a list of {@link ItemItemSimilarity}s.</p>
   *
   * <p>It's valid to build a {@link GenericItemSimilarity} this way, but perhaps missing some of the point of an
   * item-based recommender. Item-based recommenders use the assumption that item-item similarities are relatively
   * fixed, and might be known already independent of user preferences. Hence it is useful to inject that information,
   * using {@link #GenericItemSimilarity(Iterable)}.</p>
   *
   * @param otherSimilarity other {@link ItemSimilarity} to get similarities from
   * @param dataModel       data model to get items from
   * @throws TasteException if an error occurs while accessing the {@link DataModel} items
   */
  public GenericItemSimilarity(ItemSimilarity otherSimilarity, DataModel dataModel) throws TasteException {
    List<Comparable<?>> itemIDs = IteratorUtils.iterableToList(dataModel.getItemIDs());
    Iterator<ItemItemSimilarity> it = new DataModelSimilaritiesIterator(otherSimilarity, itemIDs);
    initSimilarityMaps(new IteratorIterable<ItemItemSimilarity>(it));
  }

  /**
   * <p>Like {@link #GenericItemSimilarity(ItemSimilarity, DataModel)} )}, but will only keep the specified number of
   * similarities from the given {@link DataModel}. It will keep those with the highest similarity -- those that are
   * therefore most important.</p>
   *
   * <p>Thanks to tsmorton for suggesting this and providing part of the implementation.</p>
   *
   * @param otherSimilarity other {@link ItemSimilarity} to get similarities from
   * @param dataModel       data model to get items from
   * @param maxToKeep       maximum number of similarities to keep
   * @throws TasteException if an error occurs while accessing the {@link DataModel} items
   */
  public GenericItemSimilarity(ItemSimilarity otherSimilarity, DataModel dataModel, int maxToKeep)
      throws TasteException {
    List<Comparable<?>> itemIDs = IteratorUtils.iterableToList(dataModel.getItemIDs());
    Iterator<ItemItemSimilarity> it = new DataModelSimilaritiesIterator(otherSimilarity, itemIDs);
    Iterable<ItemItemSimilarity> keptSimilarities =
        TopItems.getTopItemItemSimilarities(maxToKeep, new IteratorIterable<ItemItemSimilarity>(it));
    initSimilarityMaps(keptSimilarities);
  }

  private void initSimilarityMaps(Iterable<ItemItemSimilarity> similarities) {
    for (ItemItemSimilarity iic : similarities) {
      Comparable<?> similarityItemID1 = iic.getItemID1();
      Comparable<?> similarityItemID2 = iic.getItemID2();
      int compare = ((Comparable<Object>) similarityItemID1).compareTo(similarityItemID2);
      if (compare != 0) {
        // Order them -- first key should be the "smaller" one
        Comparable<?> itemID1;
        Comparable<?> itemID2;
        if (compare < 0) {
          itemID1 = similarityItemID1;
          itemID2 = similarityItemID2;
        } else {
          itemID1 = similarityItemID2;
          itemID2 = similarityItemID1;
        }
        Map<Comparable<?>, Double> map = similarityMaps.get(itemID1);
        if (map == null) {
          map = new FastMap<Comparable<?>, Double>();
          similarityMaps.put(itemID1, map);
        }
        map.put(itemID2, iic.getValue());
      }
      // else similarity between item and itself already assumed to be 1.0
    }
  }

  /**
   * <p>Returns the similarity between two items. Note that similarity is assumed to be symmetric, that
   * <code>itemSimilarity(item1, item2) == itemSimilarity(item2, item1)</code>, and that <code>itemSimilarity(item1,
   * item1) == 1.0</code> for all items.</p>
   *
   * @param itemID1 first item
   * @param itemID2 second item
   * @return similarity between the two
   */
  @Override
  public double itemSimilarity(Comparable<?> itemID1, Comparable<?> itemID2) {
    int compare = ((Comparable<Object>) itemID1).compareTo(itemID2);
    if (compare == 0) {
      return 1.0;
    }
    Comparable<?> firstID;
    Comparable<?> secondID;
    if (compare < 0) {
      firstID = itemID1;
      secondID = itemID2;
    } else {
      firstID = itemID2;
      secondID = itemID1;
    }
    Map<Comparable<?>, Double> nextMap = similarityMaps.get(firstID);
    if (nextMap == null) {
      return Double.NaN;
    }
    Double similarity = nextMap.get(secondID);
    return similarity == null ? Double.NaN : similarity;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // Do nothing
  }

  /** Encapsulates a similarity between two items. Similarity must be in the range [-1.0,1.0]. */
  public static final class ItemItemSimilarity implements Comparable<ItemItemSimilarity> {

    private final Comparable<?> itemID1;
    private final Comparable<?> itemID2;
    private final double value;

    /**
     * @param itemID1 first item
     * @param itemID2 second item
     * @param value similarity between the two
     * @throws IllegalArgumentException if value is NaN, less than -1.0 or greater than 1.0
     */
    public ItemItemSimilarity(Comparable<?> itemID1, Comparable<?> itemID2, double value) {
      if (itemID1 == null || itemID2 == null) {
        throw new IllegalArgumentException("An item is null");
      }
      if (Double.isNaN(value) || value < -1.0 || value > 1.0) {
        throw new IllegalArgumentException("Illegal value: " + value);
      }
      this.itemID1 = itemID1;
      this.itemID2 = itemID2;
      this.value = value;
    }

    public Comparable<?> getItemID1() {
      return itemID1;
    }

    public Comparable<?> getItemID2() {
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
      double otherValue = other.value;
      return value > otherValue ? -1 : value < otherValue ? 1 : 0;
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof ItemItemSimilarity)) {
        return false;
      }
      ItemItemSimilarity otherSimilarity = (ItemItemSimilarity) other;
      return otherSimilarity.itemID1.equals(itemID1) && otherSimilarity.itemID2.equals(itemID2) && otherSimilarity.value == value;
    }

    @Override
    public int hashCode() {
      return itemID1.hashCode() ^ itemID2.hashCode() ^ RandomUtils.hashDouble(value);
    }

  }

  private static final class DataModelSimilaritiesIterator implements Iterator<ItemItemSimilarity> {

    private final ItemSimilarity otherSimilarity;
    private final List<Comparable<?>> itemIDs;
    private final int size;
    private int i;
    private Comparable<?> itemID1;
    private int j;

    private DataModelSimilaritiesIterator(ItemSimilarity otherSimilarity, List<Comparable<?>> itemIDs) {
      this.otherSimilarity = otherSimilarity;
      this.itemIDs = itemIDs;
      this.size = itemIDs.size();
      i = 0;
      itemID1 = itemIDs.get(0);
      j = 1;
    }

    @Override
    public boolean hasNext() {
      return i < size - 1;
    }

    @Override
    public ItemItemSimilarity next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Comparable<?> itemID2 = itemIDs.get(j);
      double similarity;
      try {
        similarity = otherSimilarity.itemSimilarity(itemID1, itemID2);
      } catch (TasteException te) {
        // ugly:
        throw new RuntimeException(te);
      }
      ItemItemSimilarity result = new ItemItemSimilarity(itemID1, itemID2, similarity);
      j++;
      if (j == size) {
        i++;
        itemID1 = itemIDs.get(i);
        j = i + 1;
      }
      return result;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }

  }

}
