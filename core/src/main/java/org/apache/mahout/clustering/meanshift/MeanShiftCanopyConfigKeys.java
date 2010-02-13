package org.apache.mahout.clustering.meanshift;

public interface MeanShiftCanopyConfigKeys {
  
  // keys used by Driver, Mapper, Combiner & Reducer
  String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.canopy.measure";
  String T1_KEY = "org.apache.mahout.clustering.canopy.t1";
  String T2_KEY = "org.apache.mahout.clustering.canopy.t2";
  String CONTROL_PATH_KEY = "org.apache.mahout.clustering.control.path";
  String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.canopy.convergence";
  
}
