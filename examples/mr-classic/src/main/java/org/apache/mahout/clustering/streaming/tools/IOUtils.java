/*
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

package org.apache.mahout.clustering.streaming.tools;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.streaming.mapreduce.CentroidWritable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class IOUtils {

  private IOUtils() {}

  /**
   * Converts CentroidWritable values in a sequence file into Centroids lazily.
   * @param dirIterable the source iterable (comes from a SequenceFileDirIterable).
   * @return an Iterable<Centroid> with the converted vectors.
   */
  public static Iterable<Centroid> getCentroidsFromCentroidWritableIterable(
      Iterable<CentroidWritable>  dirIterable) {
    return Iterables.transform(dirIterable, new Function<CentroidWritable, Centroid>() {
      @Override
      public Centroid apply(CentroidWritable input) {
        Preconditions.checkNotNull(input);
        return input.getCentroid().clone();
      }
    });
  }

  /**
   * Converts CentroidWritable values in a sequence file into Centroids lazily.
   * @param dirIterable the source iterable (comes from a SequenceFileDirIterable).
   * @return an Iterable<Centroid> with the converted vectors.
   */
  public static Iterable<Centroid> getCentroidsFromClusterWritableIterable(Iterable<ClusterWritable>  dirIterable) {
    return Iterables.transform(dirIterable, new Function<ClusterWritable, Centroid>() {
      int numClusters = 0;
      @Override
      public Centroid apply(ClusterWritable input) {
        Preconditions.checkNotNull(input);
        return new Centroid(numClusters++, input.getValue().getCenter().clone(),
            input.getValue().getTotalObservations());
      }
    });
  }

  /**
   * Converts VectorWritable values in a sequence file into Vectors lazily.
   * @param dirIterable the source iterable (comes from a SequenceFileDirIterable).
   * @return an Iterable<Vector> with the converted vectors.
   */
  public static Iterable<Vector> getVectorsFromVectorWritableIterable(Iterable<VectorWritable> dirIterable) {
    return Iterables.transform(dirIterable, new Function<VectorWritable, Vector>() {
      @Override
      public Vector apply(VectorWritable input) {
        Preconditions.checkNotNull(input);
        return input.get().clone();
      }
    });
  }
}
