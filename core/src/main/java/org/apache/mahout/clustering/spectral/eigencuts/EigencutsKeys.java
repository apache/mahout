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

package org.apache.mahout.clustering.spectral.eigencuts;

/**
 * Configuration keys for the Eigencuts algorithm (analogous to KMeansConfigKeys)
 */
public final class EigencutsKeys {

  private EigencutsKeys() {}

  /**
   * B_0, or the user-specified minimum eigenflow half-life threshold
   * for an eigenvector/eigenvalue pair to be considered. Increasing
   * B_0 equates to fewer clusters
   */
  public static final String BETA = "org.apache.mahout.clustering.spectral.beta";

  /**
   * Tau, or the user-specified threshold for making cuts (setting edge
   * affinities to 0) after performing non-maximal suppression on edge weight
   * sensitivies. Increasing tau equates to more edge cuts
   */
  public static final String TAU = "org.apache.mahout.clustering.spectral.tau";

  /**
   * The normalization factor for computing the cut threshold
   */
  public static final String DELTA = "org.apache.mahout.clustering.spectral.delta";

  /**
   * Epsilon, or the user-specified coefficient that works in tandem with
   * MINIMUM_HALF_LIFE to determine which eigenvector/eigenvalue pairs to use.
   * Increasing epsilon equates to fewer eigenvector/eigenvalue pairs
   */
  public static final String EPSILON = "org.apache.mahout.clustering.spectral.epsilon";

  /**
   * Base path to the location on HDFS where the diagonal matrix (a vector)
   * and the list of eigenvalues will be stored for one of the map/reduce
   * jobs in Eigencuts.
   */
  public static final String VECTOR_CACHE_BASE = "org.apache.mahout.clustering.spectral.eigencuts.vectorcache";

  /**
   * Refers to the dimensions of the raw affinity matrix input. Since this
   * matrix is symmetrical, it is a square matrix, hence all its dimensions
   * are equal.
   */
  public static final String AFFINITY_DIMENSIONS = "org.apache.mahout.clustering.spectral.eigencuts.affinitydimensions";

  /**
   * Refers to the Path to the SequenceFile representing the affinity matrix
   */
  public static final String AFFINITY_PATH = "org.apache.mahout.clustering.spectral.eigencuts.affinitypath";

  /**
   * Refers to the Path to the SequenceFile representing the cut matrix
   */
  public static final String CUTMATRIX_PATH = "org.apache.mahout.clustering.spectral.eigencuts.cutmatrixpath";

  /**
   * Sets the SequenceFile index for the list of eigenvalues.
   */
  public static final int EIGENVALUES_CACHE_INDEX = 0;

  /**
   * Sets the SequenceFile index for the diagonal matrix.
   */
  public static final int DIAGONAL_CACHE_INDEX = 1;
}
