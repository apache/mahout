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
