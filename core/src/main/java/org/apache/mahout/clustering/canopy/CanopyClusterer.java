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

package org.apache.mahout.clustering.canopy;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

public class CanopyClusterer {

  private static final Logger log = LoggerFactory.getLogger(CanopyClusterer.class);

  private int nextCanopyId;

  // the T1 distance threshold
  private double t1;

  // the T2 distance threshold
  private double t2;

  // the T3 distance threshold
  private double t3;

  // the T4 distance threshold
  private double t4;

  // the distance measure
  private DistanceMeasure measure;

  public CanopyClusterer(DistanceMeasure measure, double t1, double t2) {
    this.t1 = t1;
    this.t2 = t2;
    this.t3 = t1;
    this.t4 = t2;
    this.measure = measure;
  }

  public double getT1() {
    return t1;
  }

  public double getT2() {
    return t2;
  }

  public double getT3() {
    return t3;
  }

  public double getT4() {
    return t4;
  }

  public CanopyClusterer(Configuration config) {
    this.configure(config);
  }

  /**
   * Configure the Canopy and its distance measure
   * 
   * @param configuration
   *            the Configuration
   */
  public void configure(Configuration configuration) {
    measure = ClassUtils.instantiateAs(configuration.get(CanopyConfigKeys.DISTANCE_MEASURE_KEY),
                                       DistanceMeasure.class);
    measure.configure(configuration);
    t1 = Double.parseDouble(configuration.get(CanopyConfigKeys.T1_KEY));
    t2 = Double.parseDouble(configuration.get(CanopyConfigKeys.T2_KEY));
    t3 = t1;
    String d = configuration.get(CanopyConfigKeys.T3_KEY);
    if (d != null) {
      t3 = Double.parseDouble(d);
    }
    t4 = t2;
    d = configuration.get(CanopyConfigKeys.T4_KEY);
    if (d != null) {
      t4 = Double.parseDouble(d);
    }
    nextCanopyId = 0;
  }

  /**
   * Used by CanopyReducer to set t1=t3 and t2=t4 configuration values
   */
  public void useT3T4() {
    t1 = t3;
    t2 = t4;
  }

  /**
   * Configure the Canopy for unit tests
   * 
   * @param aMeasure
   *            the DistanceMeasure
   * @param aT1
   *            the T1 distance threshold
   * @param aT2
   *            the T2 distance threshold
   * */
  public void config(DistanceMeasure aMeasure, double aT1, double aT2) {
    measure = aMeasure;
    t1 = aT1;
    t2 = aT2;
    t3 = t1;
    t4 = t2;
  }

  /**
   * This is the same algorithm as the reference but inverted to iterate over
   * existing canopies instead of the points. Because of this it does not need
   * to actually store the points, instead storing a total points vector and
   * the number of points. From this a centroid can be computed.
   * <p/>
   * This method is used by the CanopyMapper, CanopyReducer and CanopyDriver.
   * 
   * @param point
   *            the point to be added
   * @param canopies
   *            the List<Canopy> to be appended
   */
  public void addPointToCanopies(Vector point, Collection<Canopy> canopies) {
    boolean pointStronglyBound = false;
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(), canopy.getCenter(), point);
      if (dist < t1) {
        if (log.isDebugEnabled()) {
          log.debug("Added point: {} to canopy: {}", AbstractCluster.formatVector(point, null), canopy.getIdentifier());
        }
        canopy.observe(point);
      }
      pointStronglyBound = pointStronglyBound || dist < t2;
    }
    if (!pointStronglyBound) {
      if (log.isDebugEnabled()) {
        log.debug("Created new Canopy:{} at center:{}", nextCanopyId, AbstractCluster.formatVector(point, null));
      }
      canopies.add(new Canopy(point, nextCanopyId++, measure));
    }
  }

  /**
   * Return if the point is covered by the canopy
   * 
   * @param point
   *            a point
   * @return if the point is covered
   */
  public boolean canopyCovers(Canopy canopy, Vector point) {
    return measure.distance(canopy.getCenter().getLengthSquared(), canopy.getCenter(), point) < t1;
  }

  /**
   * Iterate through the points, adding new canopies. Return the canopies.
   * 
   * @param points
   *            a list<Vector> defining the points to be clustered
   * @param measure
   *            a DistanceMeasure to use
   * @param t1
   *            the T1 distance threshold
   * @param t2
   *            the T2 distance threshold
   * @return the List<Canopy> created
   */
  public static List<Canopy> createCanopies(List<Vector> points,
                                            DistanceMeasure measure,
                                            double t1,
                                            double t2) {
    List<Canopy> canopies = Lists.newArrayList();
    /**
     * Reference Implementation: Given a distance metric, one can create
     * canopies as follows: Start with a list of the data points in any
     * order, and with two distance thresholds, T1 and T2, where T1 > T2.
     * (These thresholds can be set by the user, or selected by
     * cross-validation.) Pick a point on the list and measure its distance
     * to all other points. Put all points that are within distance
     * threshold T1 into a canopy. Remove from the list all points that are
     * within distance threshold T2. Repeat until the list is empty.
     */
    int nextCanopyId = 0;
    while (!points.isEmpty()) {
      Iterator<Vector> ptIter = points.iterator();
      Vector p1 = ptIter.next();
      ptIter.remove();
      Canopy canopy = new Canopy(p1, nextCanopyId++, measure);
      canopies.add(canopy);
      while (ptIter.hasNext()) {
        Vector p2 = ptIter.next();
        double dist = measure.distance(p1, p2);
        // Put all points that are within distance threshold T1 into the
        // canopy
        if (dist < t1) {
          canopy.observe(p2);
        }
        // Remove from the list all points that are within distance
        // threshold T2
        if (dist < t2) {
          ptIter.remove();
        }
      }
      for (Canopy c : canopies) {
        c.computeParameters();
      }
    }
    return canopies;
  }

  /**
   * Iterate through the canopies, adding their centroids to a list
   * 
   * @param canopies
   *            a List<Canopy>
   * @return the List<Vector>
   */
  public static List<Vector> getCenters(Iterable<Canopy> canopies) {
    List<Vector> result = Lists.newArrayList();
    for (Canopy canopy : canopies) {
      result.add(canopy.getCenter());
    }
    return result;
  }

  /**
   * Iterate through the canopies, resetting their center to their centroids
   * 
   * @param canopies
   *            a List<Canopy>
   */
  public static void updateCentroids(Iterable<Canopy> canopies) {
    for (Canopy canopy : canopies) {
      canopy.computeParameters();
    }
  }

}
