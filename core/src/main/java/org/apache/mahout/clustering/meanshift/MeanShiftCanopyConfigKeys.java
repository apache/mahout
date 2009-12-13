package org.apache.mahout.clustering.meanshift;

public interface MeanShiftCanopyConfigKeys {

  // keys used by Driver, Mapper, Combiner & Reducer
  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.canopy.measure";
  public static final String T1_KEY = "org.apache.mahout.clustering.canopy.t1";
  public static final String T2_KEY = "org.apache.mahout.clustering.canopy.t2";
  public static final String CONTROL_PATH_KEY = "org.apache.mahout.clustering.control.path";
  public static final String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.canopy.convergence";

}
