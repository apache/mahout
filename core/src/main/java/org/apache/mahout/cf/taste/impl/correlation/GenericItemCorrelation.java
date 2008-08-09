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

package org.apache.mahout.cf.taste.impl.correlation;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.correlation.ItemCorrelation;
import org.apache.mahout.cf.taste.impl.common.IteratorIterable;
import org.apache.mahout.cf.taste.impl.common.IteratorUtils;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;

import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Collection;

/**
 * <p>A "generic" {@link ItemCorrelation} which takes a static list of precomputed {@link Item}
 * correlations and bases its responses on that alone. The values may have been precomputed
 * offline by another process, stored in a file, and then read and fed into an instance of this class.</p>
 *
 * <p>This is perhaps the best {@link ItemCorrelation} to use with
 * {@link org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender}, for now, since the point of item-based
 * recommenders is that they can take advantage of the fact that item similarity is relatively static,
 * can be precomputed, and then used in computation to gain a significant performance advantage.</p>
 */
public final class GenericItemCorrelation implements ItemCorrelation {

  private final Map<Item, Map<Item, Double>> correlationMaps = new FastMap<Item, Map<Item, Double>>();

  /**
   * <p>Creates a {@link GenericItemCorrelation} from a precomputed list of {@link ItemItemCorrelation}s. Each
   * represents the correlation between two distinct items. Since correlation is assumed to be symmetric,
   * it is not necessary to specify correlation between item1 and item2, and item2 and item1. Both are the same.
   * It is also not necessary to specify a correlation between any item and itself; these are assumed to be 1.0.</p>
   *
   * <p>Note that specifying a correlation between two items twice is not an error, but, the later value will
   * win.</p>
   *
   * @param correlations set of {@link ItemItemCorrelation}s on which to base this instance
   */
  public GenericItemCorrelation(Iterable<ItemItemCorrelation> correlations) {
    initCorrelationMaps(correlations);
  }

  /**
   * <p>Like {@link #GenericItemCorrelation(Iterable)}, but will only keep the specified number of correlations
   * from the given {@link Iterable} of correlations. It will keep those with the highest correlation --
   * those that are therefore most important.</p>
   *
   * <p>Thanks to tsmorton for suggesting this and providing part of the implementation.</p>
   *
   * @param correlations set of {@link ItemItemCorrelation}s on which to base this instance
   * @param maxToKeep maximum number of correlations to keep
   */
  public GenericItemCorrelation(Iterable<ItemItemCorrelation> correlations, int maxToKeep) {
    Iterable<ItemItemCorrelation> keptCorrelations = TopItems.getTopItemItemCorrelations(maxToKeep, correlations);
    initCorrelationMaps(keptCorrelations);
  }

  /**
   * <p>Builds a list of item-item correlations given an {@link ItemCorrelation} implementation and a
   * {@link DataModel}, rather than a list of {@link ItemItemCorrelation}s.</p>
   *
   * <p>It's valid to build a {@link GenericItemCorrelation} this way, but perhaps missing some of the point
   * of an item-based recommender. Item-based recommenders use the assumption that item-item correlations
   * are relatively fixed, and might be known already independent of user preferences. Hence it is useful
   * to inject that information, using {@link #GenericItemCorrelation(Iterable)}.</p>
   *
   * @param otherCorrelation other {@link ItemCorrelation} to get correlations from
   * @param dataModel data model to get {@link Item}s from
   * @throws TasteException if an error occurs while accessing the {@link DataModel} items
   */
  public GenericItemCorrelation(ItemCorrelation otherCorrelation, DataModel dataModel) throws TasteException {
    List<? extends Item> items = IteratorUtils.iterableToList(dataModel.getItems());
    Iterator<ItemItemCorrelation> it = new DataModelCorrelationsIterator(otherCorrelation, items);
    initCorrelationMaps(new IteratorIterable<ItemItemCorrelation>(it));
  }

  /**
   * <p>Like {@link #GenericItemCorrelation(ItemCorrelation, DataModel)} )}, but will only
   * keep the specified number of correlations from the given {@link DataModel}.
   * It will keep those with the highest correlation -- those that are therefore most important.</p>
   *
   * <p>Thanks to tsmorton for suggesting this and providing part of the implementation.</p>
   *
   * @param otherCorrelation other {@link ItemCorrelation} to get correlations from
   * @param dataModel data model to get {@link Item}s from
   * @param maxToKeep maximum number of correlations to keep
   * @throws TasteException if an error occurs while accessing the {@link DataModel} items
   */
  public GenericItemCorrelation(ItemCorrelation otherCorrelation, DataModel dataModel, int maxToKeep)
          throws TasteException {
    List<? extends Item> items = IteratorUtils.iterableToList(dataModel.getItems());
    Iterator<ItemItemCorrelation> it = new DataModelCorrelationsIterator(otherCorrelation, items);
    Iterable<ItemItemCorrelation> keptCorrelations =
            TopItems.getTopItemItemCorrelations(maxToKeep, new IteratorIterable<ItemItemCorrelation>(it));
    initCorrelationMaps(keptCorrelations);
  }

  private void initCorrelationMaps(Iterable<ItemItemCorrelation> correlations) {
    for (ItemItemCorrelation iic : correlations) {
      Item correlationItem1 = iic.getItem1();
      Item correlationItem2 = iic.getItem2();
      int compare = correlationItem1.compareTo(correlationItem2);
      if (compare != 0) {
        // Order them -- first key should be the "smaller" one
        Item item1;
        Item item2;
        if (compare < 0) {
          item1 = correlationItem1;
          item2 = correlationItem2;
        } else {
          item1 = correlationItem2;
          item2 = correlationItem1;
        }
        Map<Item, Double> map = correlationMaps.get(item1);
        if (map == null) {
          map = new FastMap<Item, Double>();
          correlationMaps.put(item1, map);
        }
        map.put(item2, iic.getValue());
      }
      // else correlation between item and itself already assumed to be 1.0
    }
  }

  /**
   * <p>Returns the correlation between two items. Note that correlation is assumed to be symmetric, that
   * <code>itemCorrelation(item1, item2) == itemCorrelation(item2, item1)</code>, and that
   * <code>itemCorrelation(item1, item1) == 1.0</code> for all items.</p>
   *
   * @param item1 first item
   * @param item2 second item
   * @return correlation between the two
   */
  public double itemCorrelation(Item item1, Item item2) {
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
    Map<Item, Double> nextMap = correlationMaps.get(first);
    if (nextMap == null) {
      return Double.NaN;
    }
    Double correlation = nextMap.get(second);
    return correlation == null ? Double.NaN : correlation;
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // Do nothing
  }

  /**
   * Encapsulates a correlation between two items. Correlation must be in the range [-1.0,1.0].
   */
  public static final class ItemItemCorrelation {

    // Somehow I think this class should be a top-level class now.
    // But I have a love affair with inner classes.

    private final Item item1;
    private final Item item2;
    private final double value;

    /**
     * @param item1 first item
     * @param item2 second item
     * @param value correlation between the two
     * @throws IllegalArgumentException if value is NaN, less than -1.0 or greater than 1.0
     */
    public ItemItemCorrelation(Item item1, Item item2, double value) {
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
      return "ItemItemCorrelation[" + item1 + ',' + item2 + ':' + value + ']';
    }

  }

  private static final class DataModelCorrelationsIterator implements Iterator<ItemItemCorrelation> {

    private final ItemCorrelation otherCorrelation;
    private final List<? extends Item> items;
    private final int size;
    private int i;
    private Item item1;
    private int j;

    private DataModelCorrelationsIterator(ItemCorrelation otherCorrelation, List<? extends Item> items) {
      this.otherCorrelation = otherCorrelation;
      this.items = items;
      this.size = items.size();
      i = 0;
      item1 = items.get(0);
      j = 1;
    }

    public boolean hasNext() {
      return i < size - 1;
    }

    public ItemItemCorrelation next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      Item item2 = items.get(j);
      double correlation;
      try {
        correlation = otherCorrelation.itemCorrelation(item1, item2);
      } catch (TasteException te) {
        // ugly:
        throw new RuntimeException(te);
      }
      ItemItemCorrelation result = new ItemItemCorrelation(item1, item2, correlation);
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
