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

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.common.*;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.ClusteringRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.Callable;

/**
 * <p>A {@link org.apache.mahout.cf.taste.recommender.Recommender} that clusters {@link User}s, then determines
 * the clusters' top recommendations. This implementation builds clusters by repeatedly merging clusters
 * until only a certain number remain, meaning that each cluster is sort of a tree of other clusters.</p>
 *
 * <p>This {@link org.apache.mahout.cf.taste.recommender.Recommender} therefore has a few properties to note:</p>
 * <ul>
 * <li>For all {@link User}s in a cluster, recommendations will be the same</li>
 * <li>{@link #estimatePreference(Object, Object)} may well return {@link Double#NaN}; it does so when asked
 * to estimate preference for an {@link Item} for which no preference is expressed in the {@link User}s in
 * the cluster.</li>
 * </ul>
 */
public final class TreeClusteringRecommender extends AbstractRecommender implements ClusteringRecommender {

  private static final Logger log = LoggerFactory.getLogger(TreeClusteringRecommender.class);

  private final ClusterSimilarity clusterSimilarity;
  private final int numClusters;
  private final double clusteringThreshold;
  private final boolean clusteringByThreshold;
  private final double samplingPercentage;
  private Map<Object, List<RecommendedItem>> topRecsByUserID;
  private Collection<Collection<User>> allClusters;
  private Map<Object, Collection<User>> clustersByUserID;
  private boolean clustersBuilt;
  private final ReentrantLock buildClustersLock;
  private final RefreshHelper refreshHelper;

  /**
   * @param dataModel {@link DataModel} which provdes {@link User}s
   * @param clusterSimilarity {@link ClusterSimilarity} used to compute cluster similarity
   * @param numClusters desired number of clusters to create
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>numClusters</code> is
   * less than 2
   */
  public TreeClusteringRecommender(DataModel dataModel,
                                   ClusterSimilarity clusterSimilarity,
                                   int numClusters) {
    this(dataModel, clusterSimilarity, numClusters, 1.0);
  }

  /**
   * @param dataModel {@link DataModel} which provdes {@link User}s
   * @param clusterSimilarity {@link ClusterSimilarity} used to compute cluster similarity
   * @param numClusters desired number of clusters to create
   * @param samplingPercentage percentage of all cluster-cluster pairs to consider when finding
   * next-most-similar clusters. Decreasing this value from 1.0 can increase performance at the
   * cost of accuracy
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>numClusters</code> is
   * less than 2, or samplingPercentage is {@link Double#NaN} or nonpositive or greater than 1.0
   */
  public TreeClusteringRecommender(DataModel dataModel,
                                   ClusterSimilarity clusterSimilarity,
                                   int numClusters,
                                   double samplingPercentage) {
    super(dataModel);
    if (clusterSimilarity == null) {
      throw new IllegalArgumentException("clusterSimilarity is null");
    }
    if (numClusters < 2) {
      throw new IllegalArgumentException("numClusters must be at least 2");
    }
    if (Double.isNaN(samplingPercentage) || samplingPercentage <= 0.0 || samplingPercentage > 1.0) {
      throw new IllegalArgumentException("samplingPercentage is invalid: " + samplingPercentage);
    }
    this.clusterSimilarity = clusterSimilarity;
    this.numClusters = numClusters;
    this.clusteringThreshold = Double.NaN;
    this.clusteringByThreshold = false;
    this.samplingPercentage = samplingPercentage;
    this.buildClustersLock = new ReentrantLock();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      public Object call() throws TasteException {
        buildClusters();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
    refreshHelper.addDependency(clusterSimilarity);
  }

  /**
   * @param dataModel {@link DataModel} which provdes {@link User}s
   * @param clusterSimilarity {@link ClusterSimilarity} used to compute cluster similarity
   * @param clusteringThreshold clustering similarity threshold; clusters will be aggregated into larger
   * clusters until the next two nearest clusters' similarity drops below this threshold
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>clusteringThreshold</code> is
   * {@link Double#NaN}
   */
  public TreeClusteringRecommender(DataModel dataModel,
                                   ClusterSimilarity clusterSimilarity,
                                   double clusteringThreshold) {
    this(dataModel, clusterSimilarity, clusteringThreshold, 1.0);
  }

  /**
   * @param dataModel {@link DataModel} which provdes {@link User}s
   * @param clusterSimilarity {@link ClusterSimilarity} used to compute cluster similarity
   * @param clusteringThreshold clustering similarity threshold; clusters will be aggregated into larger
   * clusters until the next two nearest clusters' similarity drops below this threshold
   * @param samplingPercentage percentage of all cluster-cluster pairs to consider when finding
   * next-most-similar clusters. Decreasing this value from 1.0 can increase performance at the
   * cost of accuracy
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>clusteringThreshold</code> is
   * {@link Double#NaN}, or samplingPercentage is {@link Double#NaN} or nonpositive or greater than 1.0
   */
  public TreeClusteringRecommender(DataModel dataModel,
                                   ClusterSimilarity clusterSimilarity,
                                   double clusteringThreshold,
                                   double samplingPercentage) {
    super(dataModel);
    if (clusterSimilarity == null) {
      throw new IllegalArgumentException("clusterSimilarity is null");
    }
    if (Double.isNaN(clusteringThreshold)) {
      throw new IllegalArgumentException("clusteringThreshold must not be NaN");
    }
    if (Double.isNaN(samplingPercentage) || samplingPercentage <= 0.0 || samplingPercentage > 1.0) {
      throw new IllegalArgumentException("samplingPercentage is invalid: " + samplingPercentage);
    }
    this.clusterSimilarity = clusterSimilarity;
    this.numClusters = Integer.MIN_VALUE;
    this.clusteringThreshold = clusteringThreshold;
    this.clusteringByThreshold = true;
    this.samplingPercentage = samplingPercentage;
    this.buildClustersLock = new ReentrantLock();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      public Object call() throws TasteException {
        buildClusters();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
    refreshHelper.addDependency(clusterSimilarity);
  }

  public List<RecommendedItem> recommend(Object userID, int howMany, Rescorer<Item> rescorer)
          throws TasteException {
    if (userID == null || rescorer == null) {
      throw new IllegalArgumentException("userID or rescorer is null");
    }
    if (howMany < 1) {
      throw new IllegalArgumentException("howMany must be at least 1");
    }
    checkClustersBuilt();

    log.debug("Recommending items for user ID '{}'", userID);

    List<RecommendedItem> recommended = topRecsByUserID.get(userID);
    if (recommended == null) {
      return Collections.emptyList();
    }

    User theUser = getDataModel().getUser(userID);
    List<RecommendedItem> rescored = new ArrayList<RecommendedItem>(recommended.size());
    // Only add items the user doesn't already have a preference for.
    // And that the rescorer doesn't "reject".
    for (RecommendedItem recommendedItem : recommended) {
      Item item = recommendedItem.getItem();
      if (rescorer.isFiltered(item)) {
        continue;
      }
      if (theUser.getPreferenceFor(item.getID()) == null &&
          !Double.isNaN(rescorer.rescore(item, recommendedItem.getValue()))) {
        rescored.add(recommendedItem);
      }
    }
    Collections.sort(rescored, new ByRescoreComparator(rescorer));

    return rescored;
  }

  public double estimatePreference(Object userID, Object itemID) throws TasteException {
    if (userID == null || itemID == null) {
      throw new IllegalArgumentException("userID or itemID is null");
    }
    DataModel model = getDataModel();
    User theUser = model.getUser(userID);
    Preference actualPref = theUser.getPreferenceFor(itemID);
    if (actualPref != null) {
      return actualPref.getValue();
    }
    checkClustersBuilt();
    List<RecommendedItem> topRecsForUser = topRecsByUserID.get(userID);
    if (topRecsForUser != null) {
      for (RecommendedItem item : topRecsForUser) {
        if (itemID.equals(item.getItem().getID())) {
          return item.getValue();
        }
      }
    }
    // Hmm, we have no idea. The item is not in the user's cluster
    return Double.NaN;
  }

  public Collection<User> getCluster(Object userID) throws TasteException {
    if (userID == null) {
      throw new IllegalArgumentException("userID is null");
    }
    checkClustersBuilt();
    Collection<User> cluster = clustersByUserID.get(userID);
    if (cluster == null) {
      return Collections.emptyList();
    } else {
      return cluster;
    }
  }

  public Collection<Collection<User>> getClusters() throws TasteException {
    checkClustersBuilt();
    return allClusters;
  }

  private void checkClustersBuilt() throws TasteException {
    if (!clustersBuilt) {
      buildClusters();
    }
  }

  private void buildClusters() throws TasteException {
    try {
      buildClustersLock.lock();
      DataModel model = getDataModel();
      int numUsers = model.getNumUsers();
      if (numUsers > 0) {
        List<Collection<User>> newClusters = new ArrayList<Collection<User>>(numUsers);
        if (numUsers == 1) {
          User onlyUser = model.getUsers().iterator().next();
          newClusters.add(Collections.singleton(onlyUser));
        } else {
          // Begin with a cluster for each user:
          for (User user : model.getUsers()) {
            Collection<User> newCluster = new HashSet<User>();
            newCluster.add(user);
            newClusters.add(newCluster);
          }
          findClusters(newClusters);
        }
        topRecsByUserID = computeTopRecsPerUserID(newClusters);
        clustersByUserID = computeClustersPerUserID(newClusters);
        allClusters = newClusters;
      } else {
        topRecsByUserID = Collections.emptyMap();
        clustersByUserID = Collections.emptyMap();
        allClusters = Collections.emptySet();
      }
      clustersBuilt = true;
    } finally {
      buildClustersLock.unlock();
    }
  }

  private void findClusters(List<Collection<User>> newClusters) throws TasteException {
    if (clusteringByThreshold) {
      Pair<Collection<User>, Collection<User>> nearestPair = findNearestClusters(newClusters);
      if (nearestPair != null) {
        Collection<User> cluster1 = nearestPair.getFirst();
        Collection<User> cluster2 = nearestPair.getSecond();
        while (clusterSimilarity.getSimilarity(cluster1, cluster2) >= clusteringThreshold) {
          newClusters.remove(cluster1);
          newClusters.remove(cluster2);
          Collection<User> merged = new HashSet<User>(cluster1.size() + cluster2.size());
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
        Pair<Collection<User>, Collection<User>> nearestPair =
                findNearestClusters(newClusters);
        if (nearestPair == null) {
          break;
        }
        Collection<User> cluster1 = nearestPair.getFirst();
        Collection<User> cluster2 = nearestPair.getSecond();
        newClusters.remove(cluster1);
        newClusters.remove(cluster2);
        Collection<User> merged = new HashSet<User>(cluster1.size() + cluster2.size());
        merged.addAll(cluster1);
        merged.addAll(cluster2);
        newClusters.add(merged);
      }
    }
  }

  private Pair<Collection<User>, Collection<User>> findNearestClusters(List<Collection<User>> clusters)
          throws TasteException {
    int size = clusters.size();
    Pair<Collection<User>, Collection<User>> nearestPair = null;
    double bestSimilarity = Double.NEGATIVE_INFINITY;
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < size; i++) {
      Collection<User> cluster1 = clusters.get(i);
      for (int j = i + 1; j < size; j++) {
        if (samplingPercentage >= 1.0 || r.nextDouble() < samplingPercentage) {
          Collection<User> cluster2 = clusters.get(j);
          double similarity = clusterSimilarity.getSimilarity(cluster1, cluster2);
          if (!Double.isNaN(similarity) && similarity > bestSimilarity) {
            bestSimilarity = similarity;
            nearestPair = new Pair<Collection<User>, Collection<User>>(cluster1, cluster2);
          }
        }
      }
    }
    return nearestPair;
  }

  private static Map<Object, List<RecommendedItem>> computeTopRecsPerUserID(
          Iterable<Collection<User>> clusters) throws TasteException {
    Map<Object, List<RecommendedItem>> recsPerUser = new FastMap<Object, List<RecommendedItem>>();
    for (Collection<User> cluster : clusters) {
      List<RecommendedItem> recs = computeTopRecsForCluster(cluster);
      for (User user : cluster) {
        recsPerUser.put(user.getID(), recs);
      }
    }
    return Collections.unmodifiableMap(recsPerUser);
  }

  private static List<RecommendedItem> computeTopRecsForCluster(Collection<User> cluster)
          throws TasteException {

    Collection<Item> allItems = new HashSet<Item>();
    for (User user : cluster) {
      Preference[] prefs = user.getPreferencesAsArray();
      for (int i = 0; i < prefs.length; i++) {
        allItems.add(prefs[i].getItem());
      }
    }

    TopItems.Estimator<Item> estimator = new Estimator(cluster);

    List<RecommendedItem> topItems =
            TopItems.getTopItems(Integer.MAX_VALUE, allItems, NullRescorer.getItemInstance(), estimator);

    log.debug("Recommendations are: {}", topItems);
    return Collections.unmodifiableList(topItems);
  }

  private static Map<Object, Collection<User>> computeClustersPerUserID(Collection<Collection<User>> clusters) {
    Map<Object, Collection<User>> clustersPerUser = new FastMap<Object, Collection<User>>(clusters.size());
    for (Collection<User> cluster : clusters) {
      for (User user : cluster) {
        clustersPerUser.put(user.getID(), cluster);
      }
    }
    return clustersPerUser;
  }

  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "TreeClusteringRecommender[clusterSimilarity:" + clusterSimilarity + ']';
  }

  private static class Estimator implements TopItems.Estimator<Item> {

    private final Collection<User> cluster;

    private Estimator(Collection<User> cluster) {
      this.cluster = cluster;
    }

    public double estimate(Item item) {
      RunningAverage average = new FullRunningAverage();
      for (User user : cluster) {
        Preference pref = user.getPreferenceFor(item.getID());
        if (pref != null) {
          average.addDatum(pref.getValue());
        }
      }
      return average.getAverage();
    }
  }
}
