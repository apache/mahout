package org.apache.mahout.clustering.kmeans;

/**
 * This class holds all config keys that are relevant to be used in the KMeans MapReduce JobConf.
 * */
public class KMeansConfigKeys {
  /** Configuration key for distance measure to use. */
  public static final String DISTANCE_MEASURE_KEY = "org.apache.mahout.clustering.kmeans.measure";
  /** Configuration key for convergence threshold. */
  public static final String CLUSTER_CONVERGENCE_KEY = "org.apache.mahout.clustering.kmeans.convergence";
  /** Configuration key for ?? */
  public static final String CLUSTER_PATH_KEY = "org.apache.mahout.clustering.kmeans.path";
  /** The number of iterations that have taken place */
  public static final String ITERATION_NUMBER = "org.apache.mahout.clustering.kmeans.iteration";
  
}
