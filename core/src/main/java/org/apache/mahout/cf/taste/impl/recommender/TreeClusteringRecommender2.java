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
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.concurrent.Callable;
import java.util.concurrent.locks.ReentrantLock;

/**
 * <p>A {@link org.apache.mahout.cf.taste.recommender.Recommender} that clusters users,
 * then determines the clusters' top recommendations. This implementation
 * builds clusters by repeatedly merging clusters until only a certain number remain, meaning that each cluster is sort
 * of a tree of other clusters.</p>
 *
 * <p>This {@link org.apache.mahout.cf.taste.recommender.Recommender} therefore has a few properties to note:</p> <ul>
 * <li>For all users in a cluster, recommendations will be the same</li>
 * <li>{@link #estimatePreference(long, long)} may well return {@link Double#NaN}; it does so
 * when asked to estimate preference for an item for which no preference is expressed in the
 * users in the cluster.</li> </ul>
 *
 * <p>This is an <em>experimental</em> implementation which tries to gain a lot of speed at the cost of accuracy in
 * building clusters, compared to {@link TreeClusteringRecommender}. It will
 * sometimes cluster two other clusters together that may not be the exact closest two clusters in existence. This may
 * not affect the recommendation quality much, but it potentially speeds up the clustering process dramatically.</p>
 */
public final class TreeClusteringRecommender2 extends AbstractRecommender implements ClusteringRecommender {

  private static final Logger log = LoggerFactory.getLogger(TreeClusteringRecommender2.class);

  private final ClusterSimilarity clusterSimilarity;
  private final int numClusters;
  private final double clusteringThreshold;
  private final boolean clusteringByThreshold;
  private FastByIDMap<List<RecommendedItem>> topRecsByUserID;
  private FastIDSet[] allClusters;
  private FastByIDMap<FastIDSet> clustersByUserID;
  private boolean clustersBuilt;
  private final ReentrantLock buildClustersLock;
  private final RefreshHelper refreshHelper;

  /**
   * @param dataModel         {@link DataModel} which provides users
   * @param clusterSimilarity {@link ClusterSimilarity} used to compute
   *                          cluster similarity
   * @param numClusters       desired number of clusters to create
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>numClusters</code> is less than 2
   */
  public TreeClusteringRecommender2(DataModel dataModel,
                                    ClusterSimilarity clusterSimilarity,
                                    int numClusters) {
    super(dataModel);
    if (clusterSimilarity == null) {
      throw new IllegalArgumentException("clusterSimilarity is null");
    }
    if (numClusters < 2) {
      throw new IllegalArgumentException("numClusters must be at least 2");
    }
    this.clusterSimilarity = clusterSimilarity;
    this.numClusters = numClusters;
    this.clusteringThreshold = Double.NaN;
    this.clusteringByThreshold = false;
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
   * @param dataModel           {@link org.apache.mahout.cf.taste.model.DataModel} which provides users
   * @param clusterSimilarity   {@link org.apache.mahout.cf.taste.impl.recommender.ClusterSimilarity} used to compute
   *                            cluster similarity
   * @param clusteringThreshold clustering similarity threshold; clusters will be aggregated into larger clusters until
   *                            the next two nearest clusters' similarity drops below this threshold
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>clusteringThreshold</code> is {@link
   *                                  Double#NaN}
   */
  public TreeClusteringRecommender2(DataModel dataModel,
                                    ClusterSimilarity clusterSimilarity,
                                    double clusteringThreshold) {
    super(dataModel);
    if (clusterSimilarity == null) {
      throw new IllegalArgumentException("clusterSimilarity is null");
    }
    if (Double.isNaN(clusteringThreshold)) {
      throw new IllegalArgumentException("clusteringThreshold must not be NaN");
    }
    this.clusterSimilarity = clusterSimilarity;
    this.numClusters = Integer.MIN_VALUE;
    this.clusteringThreshold = clusteringThreshold;
    this.clusteringByThreshold = true;
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
    Float actualPref = getDataModel().getPreferenceValue(userID, itemID);
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

  private static final class ClusterClusterPair implements Comparable<ClusterClusterPair> {

    private final FastIDSet cluster1;
    private final FastIDSet cluster2;
    private final double similarity;

    private ClusterClusterPair(FastIDSet cluster1,
                               FastIDSet cluster2,
                               double similarity) {
      this.cluster1 = cluster1;
      this.cluster2 = cluster2;
      this.similarity = similarity;
    }

    private FastIDSet getCluster1() {
      return cluster1;
    }

    private FastIDSet getCluster2() {
      return cluster2;
    }

    private double getSimilarity() {
      return similarity;
    }

    @Override
    public int hashCode() {
      return cluster1.hashCode() ^ cluster2.hashCode() ^ RandomUtils.hashDouble(similarity);
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof ClusterClusterPair)) {
        return false;
      }
      ClusterClusterPair other = (ClusterClusterPair) o;
      return cluster1.equals(other.cluster1) &&
          cluster2.equals(other.cluster2) &&
          similarity == other.similarity;
    }

    @Override
    public int compareTo(ClusterClusterPair other) {
      double otherSimilarity = other.similarity;
      if (similarity > otherSimilarity) {
        return -1;
      } else if (similarity < otherSimilarity) {
        return 1;
      } else {
        return 0;
      }
    }

  }

  private void buildClusters() throws TasteException {
    buildClustersLock.lock();
    try {
      DataModel model = getDataModel();
      int numUsers = model.getNumUsers();

      if (numUsers == 0) {

        topRecsByUserID = new FastByIDMap<List<RecommendedItem>>();
        clustersByUserID = new FastByIDMap<FastIDSet>();

      } else {

        List<FastIDSet> clusters = new LinkedList<FastIDSet>();
        // Begin with a cluster for each user:
        LongPrimitiveIterator it = model.getUserIDs();
        while (it.hasNext()) {
          FastIDSet newCluster = new FastIDSet();
          newCluster.add(it.nextLong());
          clusters.add(newCluster);
        }

        boolean done = false;
        while (!done) {
          done = mergeClosestClusters(numUsers, clusters, done);
        }

        topRecsByUserID = computeTopRecsPerUserID(clusters);
        clustersByUserID = computeClustersPerUserID(clusters);
        allClusters = clusters.toArray(new FastIDSet[clusters.size()]);

      }

      clustersBuilt = true;
    } finally {
      buildClustersLock.unlock();
    }
  }

  private boolean mergeClosestClusters(int numUsers, List<FastIDSet> clusters, boolean done)
      throws TasteException {
    // We find a certain number of closest clusters...
    LinkedList<ClusterClusterPair> queue = findClosestClusters(numUsers, clusters);

    // The first one is definitely the closest pair in existence so we can cluster
    // the two together, put it back into the set of clusters, and start again. Instead
    // we assume everything else in our list of closest cluster pairs is still pretty good,
    // and we cluster them too.

    while (!queue.isEmpty()) {

      if (!clusteringByThreshold && clusters.size() <= numClusters) {
        done = true;
        break;
      }

      ClusterClusterPair top = queue.removeFirst();

      if (clusteringByThreshold && top.getSimilarity() < clusteringThreshold) {
        done = true;
        break;
      }

      FastIDSet cluster1 = top.getCluster1();
      FastIDSet cluster2 = top.getCluster2();

      // Pull out current two clusters from clusters
      Iterator<FastIDSet> clusterIterator = clusters.iterator();
      boolean removed1 = false;
      boolean removed2 = false;
      while (clusterIterator.hasNext() && !(removed1 && removed2)) {
        FastIDSet current = clusterIterator.next();
        // Yes, use == here
        if (!removed1 && cluster1 == current) {
          clusterIterator.remove();
          removed1 = true;
        } else if (!removed2 && cluster2 == current) {
          clusterIterator.remove();
          removed2 = true;
        }
      }

      // The only catch is if a cluster showed it twice in the list of best cluster pairs;
      // have to remove the others. Pull out anything referencing these clusters from queue
      for (Iterator<ClusterClusterPair> queueIterator = queue.iterator();
           queueIterator.hasNext();) {
        ClusterClusterPair pair = queueIterator.next();
        FastIDSet pair1 = pair.getCluster1();
        FastIDSet pair2 = pair.getCluster2();
        if (pair1 == cluster1 || pair1 == cluster2 || pair2 == cluster1 || pair2 == cluster2) {
          queueIterator.remove();
        }
      }

      // Make new merged cluster
      FastIDSet merged = new FastIDSet(cluster1.size() + cluster2.size());
      merged.addAll(cluster1);
      merged.addAll(cluster2);

      // Compare against other clusters; update queue if needed
      // That new pair we're just adding might be pretty close to something else, so
      // catch that case here and put it back into our queue
      for (FastIDSet cluster : clusters) {
        double similarity = clusterSimilarity.getSimilarity(merged, cluster);
        if (similarity > queue.getLast().getSimilarity()) {
          ListIterator<ClusterClusterPair> queueIterator = queue.listIterator();
          while (queueIterator.hasNext()) {
            if (similarity > queueIterator.next().getSimilarity()) {
              queueIterator.previous();
              break;
            }
          }
          queueIterator.add(new ClusterClusterPair(merged, cluster, similarity));
        }
      }

      // Finally add new cluster to list
      clusters.add(merged);

    }
    return done;
  }

  private LinkedList<ClusterClusterPair> findClosestClusters(int numUsers, List<FastIDSet> clusters)
      throws TasteException {
    boolean full = false;
    LinkedList<ClusterClusterPair> queue = new LinkedList<ClusterClusterPair>();
    int i = 0;
    for (FastIDSet cluster1 : clusters) {
      i++;
      ListIterator<FastIDSet> it2 = clusters.listIterator(i);
      while (it2.hasNext()) {
        FastIDSet cluster2 = it2.next();
        double similarity = clusterSimilarity.getSimilarity(cluster1, cluster2);
        if (!Double.isNaN(similarity) &&
            (!full || similarity > queue.getLast().getSimilarity())) {
          ListIterator<ClusterClusterPair> queueIterator =
              queue.listIterator(queue.size());
          while (queueIterator.hasPrevious()) {
            if (similarity <= queueIterator.previous().getSimilarity()) {
              queueIterator.next();
              break;
            }
          }
          queueIterator.add(new ClusterClusterPair(cluster1, cluster2, similarity));
          if (full) {
            queue.removeLast();
          } else if (queue.size() > numUsers) { // use numUsers as queue size limit
            full = true;
            queue.removeLast();
          }
        }
      }
    }
    return queue;
  }

  private FastByIDMap<List<RecommendedItem>> computeTopRecsPerUserID(Iterable<FastIDSet> clusters) throws TasteException {
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

    List<RecommendedItem> topItems =
        TopItems.getTopItems(Integer.MAX_VALUE, allItemIDs.iterator(), null, estimator);

    log.debug("Recommendations are: {}", topItems);
    return Collections.unmodifiableList(topItems);
  }

  private static FastByIDMap<FastIDSet>
      computeClustersPerUserID(Collection<FastIDSet> clusters) {
    FastByIDMap<FastIDSet> clustersPerUser =
            new FastByIDMap<FastIDSet>(clusters.size());
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
    return "TreeClusteringRecommender2[clusterSimilarity:" + clusterSimilarity + ']';
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
