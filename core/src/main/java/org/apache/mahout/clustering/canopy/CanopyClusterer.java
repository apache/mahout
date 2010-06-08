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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class CanopyClusterer {

  private int nextCanopyId;

  private int numVectors;

  // the T1 distance threshold
  private double t1;

  // the T2 distance threshold
  private double t2;

  // the distance measure
  private DistanceMeasure measure;

  // private int nextClusterId = 0;

  public CanopyClusterer(DistanceMeasure measure, double t1, double t2) {
    this.t1 = t1;
    this.t2 = t2;
    this.measure = measure;
  }

  public CanopyClusterer(Configuration config) {
    this.configure(config);
  }

  /**
   * Configure the Canopy and its distance measure
   * 
   * @param configuration
   *          the JobConf for this job
   */
  public void configure(Configuration configuration) {
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl = ccl.loadClass(configuration.get(CanopyConfigKeys.DISTANCE_MEASURE_KEY));
      measure = (DistanceMeasure) cl.newInstance();
      measure.configure(configuration);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    t1 = Double.parseDouble(configuration.get(CanopyConfigKeys.T1_KEY));
    t2 = Double.parseDouble(configuration.get(CanopyConfigKeys.T2_KEY));
    nextCanopyId = 0;
  }

  /** Configure the Canopy for unit tests */
  public void config(DistanceMeasure aMeasure, double aT1, double aT2) {
    measure = aMeasure;
    t1 = aT1;
    t2 = aT2;
  }

  /**
   * This is the same algorithm as the reference but inverted to iterate over existing canopies instead of the
   * points. Because of this it does not need to actually store the points, instead storing a total points
   * vector and the number of points. From this a centroid can be computed.
   * <p/>
   * This method is used by the CanopyReducer.
   * 
   * @param point
   *          the point to be added
   * @param canopies
   *          the List<Canopy> to be appended
   * @param context
   *          Object to report status to the MR interface
   * @throws IOException 
   */
  public void addPointToCanopies(Vector point, List<Canopy> canopies, TaskAttemptContext context) throws IOException {
    boolean pointStronglyBound = false;
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(), canopy.getCenter(), point);
      if (dist < t1) {
        canopy.addPoint(point);
      }
      pointStronglyBound = pointStronglyBound || (dist < t2);
    }
    if (!pointStronglyBound) {
      context.setStatus("Created new Canopy:" + nextCanopyId + " numPoints:" + numVectors);
      canopies.add(new Canopy(point, nextCanopyId++));
    }
    numVectors++;
  }

  @SuppressWarnings("unchecked")
  public void emitPointToClosestCanopy(Vector point, List<Canopy> canopies, org.apache.hadoop.mapreduce.Mapper.Context context)
      throws IOException, InterruptedException {

    double minDist = Double.MAX_VALUE;
    Canopy closest = null;
    // find closest canopy
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(), canopy.getCenter(), point);
      if (dist < minDist) {
        minDist = dist;
        closest = canopy;
      }
    }
    // emit to closest canopy
    context.write(new IntWritable(closest.getId()), new WeightedVectorWritable(1, new VectorWritable(point)));
    context.setStatus("Emit Closest Canopy ID:" + closest.getIdentifier());
  }

  /**
   * Return if the point is covered by the canopy
   * 
   * @param point
   *          a point
   * @return if the point is covered
   */
  public boolean canopyCovers(Canopy canopy, Vector point) {
    return measure.distance(canopy.getCenter().getLengthSquared(), canopy.getCenter(), point) < t1;
  }

  /**
   * Iterate through the points, adding new canopies. Return the canopies.
   * 
   * @param points
   *          a list<Vector> defining the points to be clustered
   * @param measure
   *          a DistanceMeasure to use
   * @param t1
   *          the T1 distance threshold
   * @param t2
   *          the T2 distance threshold
   * @return the List<Canopy> created
   */
  public static List<Canopy> createCanopies(List<Vector> points, DistanceMeasure measure, double t1, double t2) {
    List<Canopy> canopies = new ArrayList<Canopy>();
    /**
     * Reference Implementation: Given a distance metric, one can create canopies as follows: Start with a
     * list of the data points in any order, and with two distance thresholds, T1 and T2, where T1 > T2.
     * (These thresholds can be set by the user, or selected by cross-validation.) Pick a point on the list
     * and measure its distance to all other points. Put all points that are within distance threshold T1 into
     * a canopy. Remove from the list all points that are within distance threshold T2. Repeat until the list
     * is empty.
     */
    int nextCanopyId = 0;
    while (!points.isEmpty()) {
      Iterator<Vector> ptIter = points.iterator();
      Vector p1 = ptIter.next();
      ptIter.remove();
      Canopy canopy = new Canopy(p1, nextCanopyId++);
      canopies.add(canopy);
      while (ptIter.hasNext()) {
        Vector p2 = ptIter.next();
        double dist = measure.distance(p1, p2);
        // Put all points that are within distance threshold T1 into the canopy
        if (dist < t1) {
          canopy.addPoint(p2);
        }
        // Remove from the list all points that are within distance threshold T2
        if (dist < t2) {
          ptIter.remove();
        }
      }
    }
    return canopies;
  }

  /**
   * Iterate through the canopies, adding their centroids to a list
   * 
   * @param canopies
   *          a List<Canopy>
   * @return the List<Vector>
   */
  public static List<Vector> calculateCentroids(List<Canopy> canopies) {
    List<Vector> result = new ArrayList<Vector>();
    for (Canopy canopy : canopies) {
      result.add(canopy.computeCentroid());
    }
    return result;
  }

  /**
   * Iterate through the canopies, resetting their center to their centroids
   * 
   * @param canopies
   *          a List<Canopy>
   */
  public static void updateCentroids(List<Canopy> canopies) {
    for (Canopy canopy : canopies) {
      canopy.setCenter(canopy.computeCentroid());
    }
  }

}
