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

package org.apache.mahout.clustering.streaming.mapreduce;

import java.io.IOException;

import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.FastProjectionSearch;
import org.apache.mahout.math.neighborhood.LocalitySensitiveHashSearch;
import org.apache.mahout.math.neighborhood.ProjectionSearch;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;

public final class StreamingKMeansUtilsMR {

  private StreamingKMeansUtilsMR() {
  }

  /**
   * Instantiates a searcher from a given configuration.
   * @param conf the configuration
   * @return the instantiated searcher
   * @throws RuntimeException if the distance measure class cannot be instantiated
   * @throws IllegalStateException if an unknown searcher class was requested
   */
  public static UpdatableSearcher searcherFromConfiguration(Configuration conf) {
    DistanceMeasure distanceMeasure;
    String distanceMeasureClass = conf.get(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    try {
      distanceMeasure = (DistanceMeasure) Class.forName(distanceMeasureClass).getConstructor().newInstance();
    } catch (Exception e) {
      throw new RuntimeException("Failed to instantiate distanceMeasure", e);
    }

    int numProjections =  conf.getInt(StreamingKMeansDriver.NUM_PROJECTIONS_OPTION, 20);
    int searchSize =  conf.getInt(StreamingKMeansDriver.SEARCH_SIZE_OPTION, 10);

    String searcherClass = conf.get(StreamingKMeansDriver.SEARCHER_CLASS_OPTION);

    if (searcherClass.equals(BruteSearch.class.getName())) {
      return ClassUtils.instantiateAs(searcherClass, UpdatableSearcher.class,
          new Class[]{DistanceMeasure.class}, new Object[]{distanceMeasure});
    } else if (searcherClass.equals(FastProjectionSearch.class.getName())
        || searcherClass.equals(ProjectionSearch.class.getName())) {
      return ClassUtils.instantiateAs(searcherClass, UpdatableSearcher.class,
          new Class[]{DistanceMeasure.class, int.class, int.class},
          new Object[]{distanceMeasure, numProjections, searchSize});
    } else if (searcherClass.equals(LocalitySensitiveHashSearch.class.getName())) {
      return ClassUtils.instantiateAs(searcherClass, LocalitySensitiveHashSearch.class,
          new Class[]{DistanceMeasure.class, int.class},
          new Object[]{distanceMeasure, searchSize});
    } else {
      throw new IllegalStateException("Unknown class instantiation requested");
    }
  }

  /**
   * Returns an Iterable of centroids from an Iterable of VectorWritables by creating a new Centroid containing
   * a RandomAccessSparseVector as a delegate for each VectorWritable.
   * @param inputIterable VectorWritable Iterable to get Centroids from
   * @return the new Centroids
   */
  public static Iterable<Centroid> getCentroidsFromVectorWritable(Iterable<VectorWritable> inputIterable) {
    return Iterables.transform(inputIterable, new Function<VectorWritable, Centroid>() {
      private int numVectors = 0;
      @Override
      public Centroid apply(VectorWritable input) {
        Preconditions.checkNotNull(input);
        return new Centroid(numVectors++, new RandomAccessSparseVector(input.get()), 1);
      }
    });
  }

  /**
   * Returns an Iterable of Centroid from an Iterable of Vector by either casting each Vector to Centroid (if the
   * instance extends Centroid) or create a new Centroid based on that Vector.
   * The implicit expectation is that the input will not have interleaving types of vectors. Otherwise, the numbering
   * of new Centroids will become invalid.
   * @param input Iterable of Vectors to cast
   * @return the new Centroids
   */
  public static Iterable<Centroid> castVectorsToCentroids(Iterable<Vector> input) {
    return Iterables.transform(input, new Function<Vector, Centroid>() {
      private int numVectors = 0;
      @Override
      public Centroid apply(Vector input) {
        Preconditions.checkNotNull(input);
        if (input instanceof Centroid) {
          return (Centroid) input;
        } else {
          return new Centroid(numVectors++, input, 1);
        }
      }
    });
  }

  /**
   * Writes centroids to a sequence file.
   * @param centroids the centroids to write.
   * @param path the path of the output file.
   * @param conf the configuration for the HDFS to write the file to.
   * @throws java.io.IOException
   */
  public static void writeCentroidsToSequenceFile(Iterable<Centroid> centroids, Path path, Configuration conf)
    throws IOException {
    SequenceFile.Writer writer = null;
    try {
      writer = SequenceFile.createWriter(FileSystem.get(conf), conf,
          path, IntWritable.class, CentroidWritable.class);
      int i = 0;
      for (Centroid centroid : centroids) {
        writer.append(new IntWritable(i++), new CentroidWritable(centroid));
      }
    } finally {
      Closeables.close(writer, true);
    }
  }

  public static void writeVectorsToSequenceFile(Iterable<? extends Vector> datapoints, Path path, Configuration conf)
    throws IOException {
    SequenceFile.Writer writer = null;
    try {
      writer = SequenceFile.createWriter(FileSystem.get(conf), conf,
          path, IntWritable.class, VectorWritable.class);
      int i = 0;
      for (Vector vector : datapoints) {
        writer.append(new IntWritable(i++), new VectorWritable(vector));
      }
    } finally {
      Closeables.close(writer, true);
    }
  }
}
