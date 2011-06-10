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

package org.apache.mahout.clustering.meanshift;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.kernel.IKernelProfile;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MeanShiftCanopyClusterer {
  
  private static final Logger log = LoggerFactory
      .getLogger(MeanShiftCanopyClusterer.class);
  
  private final double convergenceDelta;
  
  // the T1 distance threshold
  private final double t1;
  
  // the T2 distance threshold
  private final double t2;
  
  // the distance measure
  private final DistanceMeasure measure;
  
  private final IKernelProfile kernelProfile;
  
  public MeanShiftCanopyClusterer(Configuration configuration) {
    try {
      measure = Class
          .forName(
              configuration.get(MeanShiftCanopyConfigKeys.DISTANCE_MEASURE_KEY))
          .asSubclass(DistanceMeasure.class).newInstance();
      measure.configure(configuration);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    try {
      kernelProfile = Class
          .forName(
              configuration.get(MeanShiftCanopyConfigKeys.KERNEL_PROFILE_KEY))
          .asSubclass(IKernelProfile.class).newInstance();
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    // nextCanopyId = 0; // never read?
    t1 = Double
        .parseDouble(configuration.get(MeanShiftCanopyConfigKeys.T1_KEY));
    t2 = Double
        .parseDouble(configuration.get(MeanShiftCanopyConfigKeys.T2_KEY));
    convergenceDelta = Double.parseDouble(configuration
        .get(MeanShiftCanopyConfigKeys.CLUSTER_CONVERGENCE_KEY));
  }
  
  public MeanShiftCanopyClusterer(DistanceMeasure aMeasure,
      IKernelProfile aKernelProfileDerivative, double aT1, double aT2,
      double aDelta) {
    // nextCanopyId = 100; // so canopyIds will sort properly // never read?
    measure = aMeasure;
    t1 = aT1;
    t2 = aT2;
    convergenceDelta = aDelta;
    kernelProfile = aKernelProfileDerivative;
  }
  
  public double getT1() {
    return t1;
  }
  
  public double getT2() {
    return t2;
  }
  
  /**
   * Merge the given canopy into the canopies list. If it touches any existing
   * canopy (norm<T1) then add the center of each to the other. If it covers any
   * other canopies (norm<T2), then merge the given canopy with the closest
   * covering canopy. If the given canopy does not cover any other canopies, add
   * it to the canopies list.
   * 
   * @param aCanopy
   *          a MeanShiftCanopy to be merged
   * @param canopies
   *          the List<Canopy> to be appended
   */
  public void mergeCanopy(MeanShiftCanopy aCanopy,
      Collection<MeanShiftCanopy> canopies) {
    MeanShiftCanopy closestCoveringCanopy = null;
    double closestNorm = Double.MAX_VALUE;
    for (MeanShiftCanopy canopy : canopies) {
      double norm = measure.distance(canopy.getCenter(), aCanopy.getCenter());
      double weight = kernelProfile.calculateDerivativeValue(norm, t1);
      if (weight > 0.0) {
        aCanopy.touch(canopy, weight);
      }
      if (norm < t2 && (closestCoveringCanopy == null || norm < closestNorm)) {
        closestNorm = norm;
        closestCoveringCanopy = canopy;
      }
    }
    if (closestCoveringCanopy == null) {
      canopies.add(aCanopy);
    } else {
      closestCoveringCanopy.merge(aCanopy);
    }
  }
  
  /**
   * Shift the center to the new centroid of the cluster
   * 
   * @param canopy
   *          the canopy to shift.
   * @return if the cluster is converged
   */
  public boolean shiftToMean(MeanShiftCanopy canopy) {
    canopy.observe(canopy.getCenter(), canopy.getBoundPoints().size());
    canopy.computeConvergence(measure, convergenceDelta);
    canopy.computeParameters();
    return canopy.isConverged();
  }
  
  /**
   * Return if the point is covered by this canopy
   * 
   * @param canopy
   *          a canopy.
   * @param point
   *          a Vector point
   * @return if the point is covered
   */
  boolean covers(MeanShiftCanopy canopy, Vector point) {
    return measure.distance(canopy.getCenter(), point) < t1;
  }
  
  /**
   * Return if the point is closely covered by the canopy
   * 
   * @param canopy
   *          a canopy.
   * @param point
   *          a Vector point
   * @return if the point is covered
   */
  public boolean closelyBound(MeanShiftCanopy canopy, Vector point) {
    return measure.distance(canopy.getCenter(), point) < t2;
  }
  
  /**
   * This is the reference mean-shift implementation. Given its inputs it
   * iterates over the points and clusters until their centers converge or until
   * the maximum number of iterations is exceeded.
   * 
   * @param points
   *          the input List<Vector> of points
   * @param measure
   *          the DistanceMeasure to use
   * @param numIter
   *          the maximum number of iterations
   */
  public static List<MeanShiftCanopy> clusterPoints(Iterable<Vector> points,
      DistanceMeasure measure, IKernelProfile aKernelProfileDerivative,
      double convergenceThreshold, double t1, double t2, int numIter) {
    MeanShiftCanopyClusterer clusterer = new MeanShiftCanopyClusterer(measure,
        aKernelProfileDerivative, t1, t2, convergenceThreshold);
    int nextCanopyId = 0;
    
    List<MeanShiftCanopy> canopies = Lists.newArrayList();
    for (Vector point : points) {
      clusterer.mergeCanopy(
          new MeanShiftCanopy(point, nextCanopyId++, measure), canopies);
    }
    List<MeanShiftCanopy> newCanopies = canopies;
    boolean[] converged = {false};
    for (int iter = 0; !converged[0] && iter < numIter; iter++) {
      newCanopies = clusterer.iterate(newCanopies, converged);
    }
    return newCanopies;
  }
  
  protected List<MeanShiftCanopy> iterate(Iterable<MeanShiftCanopy> canopies,
      boolean[] converged) {
    converged[0] = true;
    List<MeanShiftCanopy> migratedCanopies = Lists.newArrayList();
    for (MeanShiftCanopy canopy : canopies) {
      converged[0] = shiftToMean(canopy) && converged[0];
      mergeCanopy(canopy, migratedCanopies);
    }
    return migratedCanopies;
  }
  
  protected static void verifyNonOverlap(Iterable<MeanShiftCanopy> canopies) {
    Collection<Integer> coveredPoints = new HashSet<Integer>();
    // verify no overlap
    for (MeanShiftCanopy canopy : canopies) {
      for (int v : canopy.getBoundPoints().toList()) {
        if (coveredPoints.contains(v)) {
          log.info("Duplicate bound point: {} in Canopy: {}", v,
              canopy.asFormatString(null));
        } else {
          coveredPoints.add(v);
        }
      }
    }
  }
  
  protected static MeanShiftCanopy findCoveringCanopy(MeanShiftCanopy canopy,
      Iterable<MeanShiftCanopy> clusters) {
    // canopies use canopyIds assigned when input vectors are processed as
    // vectorIds too
    int vectorId = canopy.getId();
    for (MeanShiftCanopy msc : clusters) {
      for (int containedId : msc.getBoundPoints().toList()) {
        if (vectorId == containedId) {
          return msc;
        }
      }
    }
    return null;
  }
  
}
