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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.Pair;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.ClusteringRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.locks.ReentrantLock;

/**
 * <p>A {@link org.apache.mahout.cf.taste.recommender.Recommender} that clusters users, then determines the
 * clusters' top recommendations. This implementation builds clusters by repeatedly merging clusters until only a
 * certain number remain, meaning that each cluster is sort of a tree of other clusters.</p>
 *
 * <p>This {@link org.apache.mahout.cf.taste.recommender.Recommender} therefore has a few properties to note:</p>
 *
 * <ul>
 * <li>For all users in a cluster, recommendations will be the same</li>
 * <li>{@link #estimatePreference(long, long)} may well return {@link Double#NaN};
 *  it does so when asked to estimate preference for an item for
 *  which no preference is expressed in the users in the cluster.</li>
 * </ul>
 */
public final class TreeClusteringRecommender extends AbstractRecommender implements ClusteringRecommender {

  private static final Logger log = LoggerFactory.getLogger(TreeClusteringRecommender.class);

  private static final FastIDSet[] NO_CLUSTERS = new FastIDSet[0];
  private static final Random r = RandomUtils.getRandom();

  private final ClusterSimilarity clusterSimilarity;
  private final int numClusters;
  private final double clusteringThreshold;
  private final boolean clusteringByThreshold;
  private final double samplingRate;
  private FastByIDMap<List<RecommendedItem>> topRecsByUserID;
  private FastIDSet[] allClusters;
  private FastByIDMap<FastIDSet> clustersByUserID;
  private boolean clustersBuilt;
  private final ReentrantLock buildClustersLock;
  private final RefreshHelper refreshHelper;

  /**
   * @param dataModel         {@link DataModel} which provdes users
   * @param clusterSimilarity {@link ClusterSimilarity} used to compute cluster similarity
   * @param numClusters       desired number of clusters to create
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>numClusters</code> is less than 2
   */
  public TreeClusteringRecommender(DataModel dataModel,
                                   ClusterSimilarity clusterSimilarity,
                                   int numClusters) {
    this(dataModel, clusterSimilarity, numClusters, 1.0);
  }

  /**
   * @param dataModel         {@link DataModel} which provdes users
   * @param clusterSimilarity {@link ClusterSimilarity} used to compute cluster similarity
   * @param numClusters       desired number of clusters to create
   * @param samplingRate      percentage of all cluster-cluster pairs to consider when finding next-most-similar
   *                          clusters. Decreasing this value from 1.0 can increase performance at the cost of accuracy
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>numClusters</code> is less than 2, or
   *                                  samplingRate is {@link Double#NaN} or nonpositive or greater than 1.0
   */
  public TreeClusteringRecommender(DataModel dataModel,
                                   ClusterSimilarity clusterSimilarity,
                                   int numClusters,
                                   double samplingRate) {
    super(dataModel);
    if (clusterSimilarity == null) {
      throw new IllegalArgumentException("clusterSimilarity is null");
    }
    if (numClusters < 2) {
      throw new IllegalArgumentException("numClusters must be at least 2");
    }
    if (Double.isNaN(samplingRate) || samplingRate <= 0.0 || samplingRate > 1.0) {
      throw new IllegalArgumentException("samplingRate is invalid: " + samplingRate);
    }
    this.clusterSimilarity = clusterSimilarity;
    this.numClusters = numClusters;
    this.clusteringThreshold = Double.NaN;
    this.clusteringByThreshold = false;
    this.samplingRate = samplingRate;
    this.buildClustersLock = new ReentrantLock();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        buildClusters();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
    refreshHelper.addDependency(clusterSimilarity);
  }

  /**
   * @param dataModel           {@link DataModel} which provdes users
   * @param clusterSimilarity   {@link ClusterSimilarity} used to compute cluster similarity
   * @param clusteringThreshold clustering similarity threshold; clusters will be aggregated into larger clusters until
   *                            the next two nearest clusters' similarity drops below this threshold
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>clusteringThreshold</code> is {@link
   *                                  Double#NaN}
   */
  public TreeClusteringRecommender(DataModel dataModel,
                                   ClusterSimilarity clusterSimilarity,
                                   double clusteringThreshold) {
    this(dataModel, clusterSimilarity, clusteringThreshold, 1.0);
  }

  /**
   * @param dataModel           {@link DataModel} which provides users
   * @param clusterSimilarity   {@link ClusterSimilarity} used to compute cluster similarity
   * @param clusteringThreshold clustering similarity threshold; clusters will be aggregated into larger clusters until
   *                            the next two nearest clusters' similarity drops below this threshold
   * @param samplingRate        percentage of all cluster-cluster pairs to consider when finding next-most-similar
   *                            clusters. Decreasing this value from 1.0 can increase performance at the cost of
   *                            accuracy
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>clusteringThreshold</code> is {@link
   *                                  Double#NaN}, or samplingRate is {@link Double#NaN} or nonpositive or greater than
   *                                  1.0
   */
  public TreeClusteringRecommender(DataModel dataModel,
                                   ClusterSimilarity clusterSimilarity,
                                   double clusteringThreshold,
                                   double samplingRate) {
    super(dataModel);
    if (clusterSimilarity == null) {
      throw new IllegalArgumentException("clusterSimilarity is null");
    }
    if (Double.isNaN(clusteringThreshold)) {
      throw new IllegalArgumentException("clusteringThreshold must not be NaN");
    }
    if (Double.isNaN(samplingRate) || samplingRate <= 0.0 || samplingRate > 1.0) {
      throw new IllegalArgumentException("samplingRate is invalid: " + samplingRate);
    }
    this.clusterSimilarity = clusterSimilarity;
    this.numClusters = Integer.MIN_VALUE;
    this.clusteringThreshold = clusteringThreshold;
    this.clusteringByThreshold = true;
    this.samplingRate = samplingRate;
    this.buildClustersLock = new ReentrantLock();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        buildClusters();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
    refreshHelper.addDependency(clusterSimilarity);
  }

  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, Rescorer<Long> rescorer)
      throws TasteException {
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }
    checkClustersBuilt();

    log.debug("Recommending items for user ID '{}'", userID);

    List<RecommendedItem> recommended = topRecsByUserID.get(userID);
    if (recommended == null) {
      return Collections.emptyList();
    }

    DataModel dataModel = getDataModel();
    List<RecommendedItem> rescored = new ArrayList<RecommendedItem>(recommended.size());
    // Only add items the user doesn't already have a preference for.
    // And that the rescorer doesn't "reject".
    for (RecommendedItem recommendedItem : recommended) {
      long itemID = recommendedItem.getItemID();
      if (rescorer != null && rescorer.isFiltered(itemID)) {
        continue;
      }
      if (dataModel.getPreferenceValue(userID, itemID) == null &&
          (rescorer == null || !Double.isNaN(rescorer.rescore(itemID, recommendedItem.getValue())))) {
        rescored.add(recommendedItem);
      }
    }
    Collections.sort(rescored, new ByRescoreComparator(rescorer));

    return rescored;
  }

  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    DataModel model = getDataModel();
    Float actualPref = model.getPreferenceValue(userID, itemID);
    if (actualPref != null) {
      return actualPref;
    }
    checkClustersBuilt();
    List<RecommendedItem> topRecsForUser = topRecsByUserID.get(userID);
    if (topRecsForUser != null) {
      for (RecommendedItem item : topRecsForUser) {
        if (itemID == item.getItemID()) {
          return item.getValue();
        }
      }
    }
    // Hmm, we have no idea. The item is not in the user's cluster
    return Float.NaN;
  }

  @Override
  public FastIDSet getCluster(long userID) throws TasteException {
    checkClustersBuilt();
    FastIDSet cluster = clustersByUserID.get(userID);
    return cluster == null ? new FastIDSet() : cluster;
  }

  @Override
  public FastIDSet[] getClusters() throws TasteException {
    checkClustersBuilt();
    return allClusters;
  }

  private void checkClustersBuilt() throws TasteException {
    if (!clustersBuilt) {
      buildClusters();
    }
  }

  private void buildClusters() throws TasteException {
    buildClustersLock.lock();
    try {
      DataModel model = getDataModel();
      int numUsers = model.getNumUsers();
      if (numUsers > 0) {
        List<FastIDSet> newClusters = new ArrayList<FastIDSet>(numUsers);
        // Begin with a cluster for each user:
        LongPrimitiveIterator it = model.getUserIDs();
        while (it.hasNext()) {
          FastIDSet newCluster = new FastIDSet();
          newCluster.add(it.nextLong());
          newClusters.add(newCluster);
        }
        if (numUsers > 1) {
          findClusters(newClusters);
        }
        topRecsByUserID = computeTopRecsPerUserID(newClusters);
        clustersByUserID = computeClustersPerUserID(newClusters);
        allClusters = newClusters.toArray(new FastIDSet[newClusters.size()]);
      } else {
        topRecsByUserID = new FastByIDMap<List<RecommendedItem>>();
        clustersByUserID = new FastByIDMap<FastIDSet>();
        allClusters = NO_CLUSTERS;
      }
      clustersBuilt = true;
    } finally {
      buildClustersLock.unlock();
    }
  }

  private void findClusters(List<FastIDSet> newClusters) throws TasteException {
    if (clusteringByThreshold) {
      Pair<FastIDSet, FastIDSet> nearestPair = findNearestClusters(newClusters);
      if (nearestPair != null) {
        FastIDSet cluster1 = nearestPair.getFirst();
        FastIDSet cluster2 = nearestPair.getSecond();
        while (clusterSimilarity.getSimilarity(cluster1, cluster2) >= clusteringThreshold) {
          newClusters.remove(cluster1);
          newClusters.remove(cluster2);
          FastIDSet merged = new FastIDSet(cluster1.size() + cluster2.size());
          merged.addAll(cluster1);
          merged.addAll(cluster2);
          newClusters.add(merged);
          nearestPair = findNearestClusters(newClusters);
          if (nearestPair == null) {
            break;
          }
          cluster1 = nearestPair.getFirst();
          cluster2 = nearestPair.getSecond();
        }
      }
    } else {
      while (newClusters.size() > numClusters) {
        Pair<FastIDSet, FastIDSet> nearestPair = findNearestClusters(newClusters);
        if (nearestPair == null) {
          break;
        }
        FastIDSet cluster1 = nearestPair.getFirst();
        FastIDSet cluster2 = nearestPair.getSecond();
        newClusters.remove(cluster1);
        newClusters.remove(cluster2);
        FastIDSet merged = new FastIDSet(cluster1.size() + cluster2.size());
        merged.addAll(cluster1);
        merged.addAll(cluster2);
        newClusters.add(merged);
      }
    }
  }

  private Pair<FastIDSet, FastIDSet> findNearestClusters(List<FastIDSet> clusters) throws TasteException {
    int size = clusters.size();
    Pair<FastIDSet, FastIDSet> nearestPair = null;
    double bestSimilarity = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < size; i++) {
      FastIDSet cluster1 = clusters.get(i);
      for (int j = i + 1; j < size; j++) {
        if (samplingRate >= 1.0 || r.nextDouble() < samplingRate) {
          FastIDSet cluster2 = clusters.get(j);
          double similarity = clusterSimilarity.getSimilarity(cluster1, cluster2);
          if (!Double.isNaN(similarity) && similarity > bestSimilarity) {
            bestSimilarity = similarity;
            nearestPair = new Pair<FastIDSet, FastIDSet>(cluster1, cluster2);
          }
        }
      }
    }
    return nearestPair;
  }

  private FastByIDMap<List<RecommendedItem>> computeTopRecsPerUserID(List<FastIDSet> clusters) throws TasteException {
    FastByIDMap<List<RecommendedItem>> recsPerUser = new FastByIDMap<List<RecommendedItem>>();
    for (FastIDSet cluster : clusters) {
      List<RecommendedItem> recs = computeTopRecsForCluster(cluster);
      LongPrimitiveIterator it = cluster.iterator();
      while (it.hasNext()) {
        recsPerUser.put(it.next(), recs);
      }
    }
    return recsPerUser;
  }

  private List<RecommendedItem> computeTopRecsForCluster(FastIDSet cluster)
      throws TasteException {
    DataModel dataModel = getDataModel();
    FastIDSet allItemIDs = new FastIDSet();
    LongPrimitiveIterator it = cluster.iterator();
    while (it.hasNext()) {
      PreferenceArray prefs = dataModel.getPreferencesFromUser(it.next());
      int size = prefs.length();
      for (int i = 0; i < size; i++) {
        allItemIDs.add(prefs.getItemID(i));
      }
    }

    TopItems.Estimator<Long> estimator = new Estimator(cluster);

    // TODO don't hardcode 100, figure out some reasonable value
    List<RecommendedItem> topItems = TopItems.getTopItems(100, allItemIDs.iterator(), null, estimator);

    log.debug("Recommendations are: {}", topItems);
    return Collections.unmodifiableList(topItems);
  }

  private static FastByIDMap<FastIDSet> computeClustersPerUserID(List<FastIDSet> clusters) {
    FastByIDMap<FastIDSet> clustersPerUser = new FastByIDMap<FastIDSet>(clusters.size());
    for (FastIDSet cluster : clusters) {
      LongPrimitiveIterator it = cluster.iterator();
      while (it.hasNext()) {
        clustersPerUser.put(it.next(), cluster);
      }
    }
    return clustersPerUser;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "TreeClusteringRecommender[clusterSimilarity:" + clusterSimilarity + ']';
  }

  private class Estimator implements TopItems.Estimator<Long> {

    private final FastIDSet cluster;

    private Estimator(FastIDSet cluster) {
      this.cluster = cluster;
    }

    @Override
    public double estimate(Long itemID) throws TasteException {
      DataModel dataModel = getDataModel();
      RunningAverage average = new FullRunningAverage();
      LongPrimitiveIterator it = cluster.iterator();
      while (it.hasNext()) {
        Float pref = dataModel.getPreferenceValue(it.next(), itemID);
        if (pref != null) {
          average.addDatum(pref);
        }
      }
      return average.getAverage();
    }
  }
}
