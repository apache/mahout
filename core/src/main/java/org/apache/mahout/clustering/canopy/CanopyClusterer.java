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
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;

public class CanopyClusterer {

  private int nextCanopyId;
  
  // the T1 distance threshold
  private double t1;

  // the T2 distance threshold
  private double t2;

  // the distance measure
  private DistanceMeasure measure;

  //private int nextClusterId = 0;
  
  public CanopyClusterer(DistanceMeasure measure, double t1, double t2) {
    this.t1 = t1;
    this.t2 = t2;
    this.measure = measure;
  }

  public CanopyClusterer(JobConf job) {
    this.configure(job);
  }

  /**
   * Configure the Canopy and its distance measure
   * 
   * @param job the JobConf for this job
   */
  public void configure(JobConf job) {
    try {
      ClassLoader ccl = Thread.currentThread().getContextClassLoader();
      Class<?> cl = ccl.loadClass(job
          .get(CanopyConfigKeys.DISTANCE_MEASURE_KEY));
      measure = (DistanceMeasure) cl.newInstance();
      measure.configure(job);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    t1 = Double.parseDouble(job.get(CanopyConfigKeys.T1_KEY));
    t2 = Double.parseDouble(job.get(CanopyConfigKeys.T2_KEY));
    nextCanopyId = 0;
  }

  /** Configure the Canopy for unit tests */
  public void config(DistanceMeasure aMeasure, double aT1, double aT2) {
    measure = aMeasure;
    t1 = aT1;
    t2 = aT2;
  }

  /**
   * This is the same algorithm as the reference but inverted to iterate over
   * existing canopies instead of the points. Because of this it does not need
   * to actually store the points, instead storing a total points vector and the
   * number of points. From this a centroid can be computed.
   * <p/>
   * This method is used by the CanopyReducer.
   * 
   * @param point the point to be added
   * @param canopies the List<Canopy> to be appended
   */
  public void addPointToCanopies(Vector point, List<Canopy> canopies) {
    boolean pointStronglyBound = false;
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(),
          canopy.getCenter(), point);
      if (dist < t1) {
        canopy.addPoint(point);
      }
      pointStronglyBound = pointStronglyBound || (dist < t2);
    }
    if (!pointStronglyBound) {
      canopies.add(new Canopy(point, nextCanopyId++));
    }
  }

  /**
   * This method is used by the CanopyMapper to perform canopy inclusion tests
   * and to emit the point and its covering canopies to the output. The
   * CanopyCombiner will then sum the canopy points and produce the centroids.
   * 
   * @param point the point to be added
   * @param canopies the List<Canopy> to be appended
   * @param collector an OutputCollector in which to emit the point
   */
  public void emitPointToNewCanopies(Vector point, List<Canopy> canopies,
      OutputCollector<Text, Vector> collector) throws IOException {
    boolean pointStronglyBound = false;
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(),
          canopy.getCenter(), point);
      if (dist < t1) {
        canopy.emitPoint(point, collector);
      }
      pointStronglyBound = pointStronglyBound || (dist < t2);
    }
    if (!pointStronglyBound) {
      Canopy canopy = new Canopy(point, nextCanopyId++);
      canopies.add(canopy);
      canopy.emitPoint(point, collector);
    }
  }

  /**
   * This method is used by the CanopyMapper to perform canopy inclusion tests
   * and to emit the point keyed by its covering canopies to the output. if the
   * point is not covered by any canopies (due to canopy centroid clustering),
   * emit the point to the closest covering canopy.
   * 
   * @param point the point to be added
   * @param canopies the List<Canopy> to be appended
   * @param collector an OutputCollector in which to emit the point
   */
  public void emitPointToExistingCanopies(Vector point, List<Canopy> canopies,
      OutputCollector<Text, Vector> collector) throws IOException {
    double minDist = Double.MAX_VALUE;
    Canopy closest = null;
    boolean isCovered = false;
    for (Canopy canopy : canopies) {
      double dist = measure.distance(canopy.getCenter().getLengthSquared(),
          canopy.getCenter(), point);
      if (dist < t1) {
        isCovered = true;
        collector.collect(new Text(canopy.getIdentifier()), point);
      } else if (dist < minDist) {
        minDist = dist;
        closest = canopy;
      }
    }
    // if the point is not contained in any canopies (due to canopy centroid
    // clustering), emit the point to the closest covering canopy.
    if (!isCovered) {
      collector.collect(new Text(closest.getIdentifier()), point);
    }
  }

  /**
   * Return if the point is covered by the canopy
   * 
   * @param point a point
   * @return if the point is covered
   */
  public boolean canopyCovers(Canopy canopy, Vector point) {
    return measure.distance(canopy.getCenter().getLengthSquared(), 
        canopy.getCenter(), point) < t1;
  }
}
