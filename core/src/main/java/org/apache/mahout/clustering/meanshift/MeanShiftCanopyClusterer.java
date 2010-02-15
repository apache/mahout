package org.apache.mahout.clustering.meanshift;

import java.io.IOException;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.Vector;

public class MeanShiftCanopyClusterer {
  
  private double convergenceDelta = 0;
  // the next canopyId to be allocated
  private int nextCanopyId = 0;
  // the T1 distance threshold
  private double t1;
  // the T2 distance threshold
  private double t2;
  // the distance measure
  private DistanceMeasure measure;
  
  public MeanShiftCanopyClusterer(JobConf job) {
    configure(job);
  }
  
  public MeanShiftCanopyClusterer(DistanceMeasure aMeasure, double aT1, double aT2, double aDelta) {
    config(aMeasure, aT1, aT2, aDelta);
  }
  
  public double getT1() {
    return t1;
  }
  
  public double getT2() {
    return t2;
  }
  
  /**
   * Configure the Canopy and its distance measure
   * 
   * @param job
   *          the JobConf for this job
   */
  public void configure(JobConf job) {
    try {
      measure = Class.forName(job.get(MeanShiftCanopyConfigKeys.DISTANCE_MEASURE_KEY)).asSubclass(
        DistanceMeasure.class).newInstance();
      measure.configure(job);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(e);
    } catch (InstantiationException e) {
      throw new IllegalStateException(e);
    }
    nextCanopyId = 0;
    t1 = Double.parseDouble(job.get(MeanShiftCanopyConfigKeys.T1_KEY));
    t2 = Double.parseDouble(job.get(MeanShiftCanopyConfigKeys.T2_KEY));
    convergenceDelta = Double.parseDouble(job.get(MeanShiftCanopyConfigKeys.CLUSTER_CONVERGENCE_KEY));
  }
  
  /**
   * Configure the Canopy for unit tests
   * 
   * @param aDelta
   *          the convergence criteria
   */
  public void config(DistanceMeasure aMeasure, double aT1, double aT2, double aDelta) {
    nextCanopyId = 100; // so canopyIds will sort properly
    measure = aMeasure;
    t1 = aT1;
    t2 = aT2;
    convergenceDelta = aDelta;
  }
  
  /**
   * Merge the given canopy into the canopies list. If it touches any existing canopy (norm<T1) then add the
   * center of each to the other. If it covers any other canopies (norm<T2), then merge the given canopy with
   * the closest covering canopy. If the given canopy does not cover any other canopies, add it to the
   * canopies list.
   * 
   * @param aCanopy
   *          a MeanShiftCanopy to be merged
   * @param canopies
   *          the List<Canopy> to be appended
   */
  public void mergeCanopy(MeanShiftCanopy aCanopy, List<MeanShiftCanopy> canopies) {
    MeanShiftCanopy closestCoveringCanopy = null;
    double closestNorm = Double.MAX_VALUE;
    for (MeanShiftCanopy canopy : canopies) {
      double norm = measure.distance(canopy.getCenter(), aCanopy.getCenter());
      if (norm < t1) {
        aCanopy.touch(canopy);
      }
      if (norm < t2) {
        if ((closestCoveringCanopy == null) || (norm < closestNorm)) {
          closestNorm = norm;
          closestCoveringCanopy = canopy;
        }
      }
    }
    if (closestCoveringCanopy == null) {
      canopies.add(aCanopy);
    } else {
      closestCoveringCanopy.merge(aCanopy);
    }
  }
  
  /** Emit the new canopy to the collector, keyed by the canopy's Id */
  static void emitCanopy(MeanShiftCanopy canopy,
                         OutputCollector<Text,WritableComparable<?>> collector) throws IOException {
    String identifier = canopy.getIdentifier();
    collector.collect(new Text(identifier), new Text("new " + canopy.toString()));
  }
  
  /**
   * Shift the center to the new centroid of the cluster
   * 
   * @param canopy
   *          the canopy to shift.
   * @return if the cluster is converged
   */
  public boolean shiftToMean(MeanShiftCanopy canopy) {
    Vector centroid = canopy.computeCentroid();
    canopy
        .setConverged(new EuclideanDistanceMeasure().distance(centroid, canopy.getCenter()) < convergenceDelta);
    canopy.setCenter(centroid);
    canopy.setNumPoints(1);
    canopy.setPointTotal(centroid.clone());
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
}
