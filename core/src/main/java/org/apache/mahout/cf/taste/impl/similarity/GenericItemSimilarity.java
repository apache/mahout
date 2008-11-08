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
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.IteratorIterable;
import org.apache.mahout.cf.taste.impl.common.IteratorUtils;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**
 * <p>A "generic" {@link ItemSimilarity} which takes a static list of precomputed {@link Item}
 * similarities and bases its responses on that alone. The values may have been precomputed
 * offline by another process, stored in a file, and then read and fed into an instance of this class.</p>
 *
 * <p>This is perhaps the best {@link ItemSimilarity} to use with
 * {@link org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender}, for now, since the point of item-based
 * recommenders is that they can take advantage of the fact that item similarity is relatively static,
 * can be precomputed, and then used in computation to gain a significant performance advantage.</p>
 */
public final class GenericItemSimilarity implements ItemSimilarity {

  private final Map<Item, Map<Item, Double>> similarityMaps = new FastMap<Item, Map<Item, Double>>();

  /**
   * <p>Creates a {@link GenericItemSimilarity} from a precomputed list of
   * {@link org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity.ItemItemSimilarity}s. Each
   * represents the similarity between two distinct items. Since similarity is assumed to be symmetric,
   * it is not necessary to specify similarity between item1 and item2, and item2 and item1. Both are the same.
   * It is also not necessary to specify a similarity between any item and itself; these are assumed to be 1.0.</p>
   *
   * <p>Note that specifying a similarity between two items twice is not an error, but, the later value will
   * win.</p>
   *
   * @param similarities set of
   *  {@link org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity.ItemItemSimilarity}s
   *  on which to base this instance
   */
  public GenericItemSimilarity(Iterable<ItemItemSimilarity> similarities) {
    initSimilarityMaps(similarities);
  }

  /**
   * <p>Like {@link #GenericItemSimilarity(Iterable)}, but will only keep the specified number of similarities
   * from the given {@link Iterable} of similarities. It will keep those with the highest similarity --
   * those that are therefore most important.</p>
   *
   * <p>Thanks to tsmorton for suggesting this and providing part of the implementation.</p>
   *
   * @param similarities set of
   *  {@link org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity.ItemItemSimilarity}s
   *  on which to base this instance
   * @param maxToKeep maximum number of similarities to keep
   */
  public GenericItemSimilarity(Iterable<ItemItemSimilarity> similarities, int maxToKeep) {
    Iterable<ItemItemSimilarity> keptSimilarities = TopItems.getTopItemItemSimilarities(maxToKeep, similarities);
    initSimilarityMaps(keptSimilarities);
  }

  /**
   * <p>Builds a list of item-item similarities given an {@link ItemSimilarity} implementation and a
   * {@link DataModel}, rather than a list of
   * {@link org.apache.mahout.cf.taste.impl.similarity.GenericItemSimilarity.ItemItemSimilarity}s.</p>
   *
   * <p>It's valid to build a {@link GenericItemSimilarity} this way, but perhaps missing some of the point
   * of an item-based recommender. Item-based recommenders use the assumption that item-item similarities
   * are relatively fixed, and might be known already independent of user preferences. Hence it is useful
   * to inject that information, using {@link #GenericItemSimilarity(Iterable)}.</p>
   *
   * @param otherSimilarity other {@link ItemSimilarity} to get similarities from
   * @param dataModel data model to get {@link Item}s from
   * @throws TasteException if an error occurs while accessing the {@link DataModel} items
   */
  public GenericItemSimilarity(ItemSimilarity otherSimilarity, DataModel dataModel) throws TasteException {
    List<? extends Item> items = IteratorUtils.iterableToList(dataModel.getItems());
    Iterator<ItemItemSimilarity> it = new DataModelSimilaritiesIterator(otherSimilarity, items);
    initSimilarityMaps(new IteratorIterable<ItemItemSimilarity>(it));
  }

  /**
   * <p>Like {@link #GenericItemSimilarity(ItemSimilarity, DataModel)} )}, but will only
   * keep the specified number of similarities from the given {@link DataModel}.
   * It will keep those with the highest similarity -- those that are therefore most important.</p>
   *
   * <p>Thanks to tsmorton for suggesting this and providing part of the implementation.</p>
   *
   * @param otherSimilarity other {@link ItemSimilarity} to get similarities from
   * @param dataModel data model to get {@link Item}s from
   * @param maxToKeep maximum number of similarities to keep
   * @throws TasteException if an error occurs while accessing the {@link DataModel} items
   */
  public GenericItemSimilarity(ItemSimilarity otherSimilarity, DataModel dataModel, int maxToKeep)
          throws TasteException {
    List<? extends Item> items = IteratorUtils.iterableToList(dataModel.getItems());
    Iterator<ItemItemSimilarity> it = new DataModelSimilaritiesIterator(otherSimilarity, items);
    Iterable<ItemItemSimilarity> keptSimilarities =
            TopItems.getTopItemItemSimilarities(maxToKeep, new IteratorIterable<ItemItemSimilarity>(it));
    initSimilarityMaps(keptSimilarities);
  }

  private void initSimilarityMaps(Iterable<ItemItemSimilarity> similarities) {
    for (ItemItemSimilarity iic : similarities) {
      Item similarityItem1 = iic.getItem1();
      Item similarityItem2 = iic.getItem2();
      int compare = similarityItem1.compareTo(similarityItem2);
      if (compare != 0) {
        // Order them -- first key should be the "smaller" one
        Item item1;
        Item item2;
        if (compare < 0) {
          item1 = similarityItem1;
          item2 = similarityItem2;
        } else {
          item1 = similarityItem2;
          item2 = similarityItem1;
        }
        Map<Item, Double> map = similarityMaps.get(item1);
        if (map == null) {
          map = new FastMap<Item, Double>();
          similarityMaps.put(item1, map);
        }
        map.put(item2, iic.getValue());
      }
      // else similarity between item and itself already assumed to be 1.0
    }
  }

  /**
   * <p>Returns the similarity between two items. Note that similarity is assumed to be symmetric, that
   * <code>itemSimilarity(item1, item2) == itemSimilarity(item2, item1)</code>, and that
   * <code>itemSimilarity(item1, item1) == 1.0</code> for all items.</p>
   *
   * @param item1 first item
   * @param item2 second item
   * @return similarity between the two
   */
  public double itemSimilarity(Item item1, Item item2) {
    int compare = item1.compareTo(item2);
    if (compare == 0) {
      return 1.0;
    }
    Item first;
    Item second;
    if (compare < 0) {
      first = item1;
      second = item2;
    } else {
      first = item2;
      second = item1;
    }
    Map<Item, Double> nextMap = similarityMaps.get(first);
    if (nextMap == null) {
      return Double.NaN;
    }
    Double similarity = nextMap.get(second);
    return similarity == null ? Double.NaN : similarity;
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // Do nothing
  }

  /**
   * Encapsulates a similarity between two items. Similarity must be in the range [-1.0,1.0].
   */
  public static final class ItemItemSimilarity implements Comparable<ItemItemSimilarity> {

    private final Item item1;
    private final Item item2;
    private final double value;

    /**
     * @param item1 first item
     * @param item2 second item
     * @param value similarity between the two
     * @throws IllegalArgumentException if value is NaN, less than -1.0 or greater than 1.0
     */
    public ItemItemSimilarity(Item item1, Item item2, double value) {
      if (item1 == null || item2 == null) {
        throw new IllegalArgumentException("An item is null");
      }
      if (Double.isNaN(value) || value < -1.0 || value > 1.0) {
        throw new IllegalArgumentException("Illegal value: " + value);
      }
      this.item1 = item1;
      this.item2 = item2;
      this.value = value;
    }

    public Item getItem1() {
      return item1;
    }

    public Item getItem2() {
      return item2;
    }

    public double getValue() {
      return value;
    }

    @Override
    public String toString() {
      return "ItemItemSimilarity[" + item1 + ',' + item2 + ':' + value + ']';
    }

    /**
     * Defines an ordering from highest similarity to lowest.
     */
    public int compareTo(ItemItemSimilarity other) {
      double otherValue = other.value;
      return value > otherValue ? -1 : value < otherValue ? 1 : 0;
    }

  }

  private static final class DataModelSimilaritiesIterator implements Iterator<ItemItemSimilarity> {

    private final ItemSimilarity otherSimilarity;
    private final List<? extends Item> items;
    private final int size;
    private int i;
    private Item item1;
    private int j;

    private DataModelSimilaritiesIterator(ItemSimilarity otherSimilarity, List<? extends Item> items) {
      this.otherSimilarity = otherSimilarity;
      this.items = items;
      this.size = items.size();
      i = 0;
      item1 = items.get(0);
      j = 1;
    }

    public boolean hasNext() {
      return i < size - 1;
    }

    public ItemItemSimilarity next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Item item2 = items.get(j);
      double similarity;
      try {
        similarity = otherSimilarity.itemSimilarity(item1, item2);
      } catch (TasteException te) {
        // ugly:
        throw new RuntimeException(te);
      }
      ItemItemSimilarity result = new ItemItemSimilarity(item1, item2, similarity);
      j++;
      if (j == size) {
        i++;
        item1 = items.get(i);
        j = i + 1;
      }
      return result;
    }

    public void remove() {
      throw new UnsupportedOperationException();
    }

  }

}
