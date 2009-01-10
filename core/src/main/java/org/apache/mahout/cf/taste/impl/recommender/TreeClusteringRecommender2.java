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
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.common.RandomUtils;
import org.apache.mahout.cf.taste.impl.common.FastSet;
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
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.locks.ReentrantLock;

/**
 * <p>A {@link org.apache.mahout.cf.taste.recommender.Recommender} that clusters
 * {@link org.apache.mahout.cf.taste.model.User}s, then determines
 * the clusters' top recommendations. This implementation builds clusters by repeatedly merging clusters
 * until only a certain number remain, meaning that each cluster is sort of a tree of other clusters.</p>
 *
 * <p>This {@link org.apache.mahout.cf.taste.recommender.Recommender} therefore has a few properties to note:</p>
 * <ul>
 * <li>For all {@link org.apache.mahout.cf.taste.model.User}s in a cluster, recommendations will be the same</li>
 * <li>{@link #estimatePreference(Object, Object)} may well return {@link Double#NaN}; it does so when asked
 * to estimate preference for an {@link org.apache.mahout.cf.taste.model.Item} for which no preference is expressed in the
 * {@link org.apache.mahout.cf.taste.model.User}s in the cluster.</li>
 * </ul>
 *
 * <p>This is an <em>experimental</em> implementation which tries to gain a lot of speed at the cost of
 * accuracy in building clusters, compared to {@link org.apache.mahout.cf.taste.impl.recommender.TreeClusteringRecommender}.
 * It will sometimes cluster two other clusters together that may not be the exact closest two clusters
 * in existence. This may not affect the recommendation quality much, but it potentially speeds up the
 * clustering process dramatically.</p>
 */
public final class TreeClusteringRecommender2 extends AbstractRecommender implements ClusteringRecommender {

  private static final Logger log = LoggerFactory.getLogger(TreeClusteringRecommender2.class);

  private final ClusterSimilarity clusterSimilarity;
  private final int numClusters;
  private final double clusteringThreshold;
  private final boolean clusteringByThreshold;
  private Map<Object, List<RecommendedItem>> topRecsByUserID;
  private Collection<Collection<User>> allClusters;
  private Map<Object, Collection<User>> clustersByUserID;
  private boolean clustersBuilt;
  private final ReentrantLock buildClustersLock;
  private final RefreshHelper refreshHelper;

  /**
   * @param dataModel {@link org.apache.mahout.cf.taste.model.DataModel} which provdes {@link org.apache.mahout.cf.taste.model.User}s
   * @param clusterSimilarity {@link org.apache.mahout.cf.taste.impl.recommender.ClusterSimilarity} used to compute
   * cluster similarity
   * @param numClusters desired number of clusters to create
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>numClusters</code> is
   * less than 2
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
   * @param dataModel {@link org.apache.mahout.cf.taste.model.DataModel} which provdes {@link org.apache.mahout.cf.taste.model.User}s
   * @param clusterSimilarity {@link org.apache.mahout.cf.taste.impl.recommender.ClusterSimilarity} used to compute
   * cluster similarity
   * @param clusteringThreshold clustering similarity threshold; clusters will be aggregated into larger
   * clusters until the next two nearest clusters' similarity drops below this threshold
   * @throws IllegalArgumentException if arguments are <code>null</code>, or <code>clusteringThreshold</code> is
   * {@link Double#NaN}
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
  public List<RecommendedItem> recommend(Object userID, int howMany, Rescorer<Item> rescorer)
          throws TasteException {
    if (userID == null) {
      throw new IllegalArgumentException("userID is null");
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
      if (rescorer != null && rescorer.isFiltered(item)) {
        continue;
      }
      if (theUser.getPreferenceFor(item.getID()) == null &&
          (rescorer == null || !Double.isNaN(rescorer.rescore(item, recommendedItem.getValue())))) {
        rescored.add(recommendedItem);
      }
    }
    Collections.sort(rescored, new ByRescoreComparator(rescorer));

    return rescored;
  }

  @Override
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

  @Override
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

  @Override
  public Collection<Collection<User>> getClusters() throws TasteException {
    checkClustersBuilt();
    return allClusters;
  }

  private void checkClustersBuilt() throws TasteException {
    if (!clustersBuilt) {
      buildClusters();
    }
  }

  private static final class ClusterClusterPair implements Comparable<ClusterClusterPair> {

    private final Collection<User> cluster1;
    private final Collection<User> cluster2;
    private final double similarity;

    private ClusterClusterPair(Collection<User> cluster1,
                               Collection<User> cluster2,
                               double similarity) {
      this.cluster1 = cluster1;
      this.cluster2 = cluster2;
      this.similarity = similarity;
    }

    private Collection<User> getCluster1() {
      return cluster1;
    }

    private Collection<User> getCluster2() {
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

        topRecsByUserID = Collections.emptyMap();
        clustersByUserID = Collections.emptyMap();

      } else {

        List<Collection<User>> clusters = new LinkedList<Collection<User>>();
        // Begin with a cluster for each user:
        for (User user : model.getUsers()) {
          Collection<User> newCluster = new FastSet<User>();
          newCluster.add(user);
          clusters.add(newCluster);
        }

        boolean done = false;
        while (!done) {
          done = mergeClosestClusters(numUsers, clusters, done);
        }

        topRecsByUserID = computeTopRecsPerUserID(clusters);
        clustersByUserID = computeClustersPerUserID(clusters);
        allClusters = clusters;

      }

      clustersBuilt = true;
    } finally {
      buildClustersLock.unlock();
    }
  }

  private boolean mergeClosestClusters(int numUsers, List<Collection<User>> clusters, boolean done) 
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

      Collection<User> cluster1 = top.getCluster1();
      Collection<User> cluster2 = top.getCluster2();

      // Pull out current two clusters from clusters
      Iterator<Collection<User>> clusterIterator = clusters.iterator();
      boolean removed1 = false;
      boolean removed2 = false;
      while (clusterIterator.hasNext() && !(removed1 && removed2)) {
        Collection<User> current = clusterIterator.next();
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
        Collection<User> pair1 = pair.getCluster1();
        Collection<User> pair2 = pair.getCluster2();
        if (pair1 == cluster1 || pair1 == cluster2 || pair2 == cluster1 || pair2 == cluster2) {
          queueIterator.remove();
        }
      }

      // Make new merged cluster
      Collection<User> merged = new FastSet<User>(cluster1.size() + cluster2.size());
      merged.addAll(cluster1);
      merged.addAll(cluster2);

      // Compare against other clusters; update queue if needed
      // That new pair we're just adding might be pretty close to something else, so
      // catch that case here and put it back into our queue
      for (Collection<User> cluster : clusters) {
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

  private LinkedList<ClusterClusterPair> findClosestClusters(int numUsers, List<Collection<User>> clusters)
      throws TasteException {
    boolean full = false;
    LinkedList<ClusterClusterPair> queue = new LinkedList<ClusterClusterPair>();
    int i = 0;
    for (Collection<User> cluster1 : clusters) {
      i++;
      ListIterator<Collection<User>> it2 = clusters.listIterator(i);
      while (it2.hasNext()) {
        Collection<User> cluster2 = it2.next();
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

  private static Map<Object, List<RecommendedItem>> computeTopRecsPerUserID(Iterable<Collection<User>> clusters)
          throws TasteException {
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

    Collection<Item> allItems = new FastSet<Item>();
    for (User user : cluster) {
      Preference[] prefs = user.getPreferencesAsArray();
      for (Preference pref : prefs) {
        allItems.add(pref.getItem());
      }
    }

    TopItems.Estimator<Item> estimator = new Estimator(cluster);

    List<RecommendedItem> topItems =
            TopItems.getTopItems(Integer.MAX_VALUE, allItems, null, estimator);

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

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

  @Override
  public String toString() {
    return "TreeClusteringRecommender2[clusterSimilarity:" + clusterSimilarity + ']';
  }

  private static class Estimator implements TopItems.Estimator<Item> {

    private final Collection<User> cluster;

    private Estimator(Collection<User> cluster) {
      this.cluster = cluster;
    }

    @Override
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
