package org.apache.mahout.clustering.kmeans;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.*;

/**
 * Given an Input Path containing a {@link org.apache.hadoop.io.SequenceFile}, select cluster seeds
 * according to Assumption-Free K-MC2 algorithm ( https://papers.nips.cc/paper/6478-fast-and-provably-good-seedings-for-k-means.pdf )
 * and write them to the output file as a {@link org.apache.mahout.clustering.kmeans.Kluster} representing the
 * initial centroid to use.
 *
 * There are some changes regarding sampling parameters and the sampling should be able to be performed in MapReduce.
 */
public final class AssumptionFreeKMCMCSeedGenerator {

  /*
   * Helper class for Weighted Random Sampling using Reservoir
   */
  private static class CandidateEntry implements Comparable {
    double priority;
    Vector coordinates;
    double weight;
    String text;

    private CandidateEntry(double priority, Vector coordinates, double weight, String text) {
      this.priority = priority;
      this.coordinates = coordinates;
      this.weight = weight;
      this.text = text;
    }

    @Override
    public int compareTo(Object o) {
      return ((Double) priority).compareTo(((CandidateEntry) o).priority);
    }

    @Override
    public boolean equals(Object o) {
      return this.coordinates == ((CandidateEntry) o).coordinates;
    }
  }

  private static final int DEFAULT_M = 100;  // Seems like a good trade-off between

  private AssumptionFreeKMCMCSeedGenerator() {
  }

  public static Path buildSeed(Configuration conf, Path input, Path output, int k, DistanceMeasure measure) throws IOException {
    return buildSeed(conf, input, output, k, DEFAULT_M, measure, null);
  }

  public static Path buildSeed(Configuration conf, Path input, Path output, int k, int m, DistanceMeasure measure) throws IOException {
    return buildSeed(conf, input, output, k, m, measure, null);
  }

  public static Path buildSeed(Configuration conf, Path input, Path output, int k, int m, DistanceMeasure measure, Long seed) throws IOException {
    Preconditions.checkArgument(k > 0, "Must be: k > 0, but k = " + k);
    Preconditions.checkArgument(m > 0, "Must be: m > 0, but m = " + m);
    FileSystem fs = FileSystem.get(output.toUri(), conf);
    HadoopUtil.delete(conf, output);

    Path outFile = new Path(output, "part-AFKMC2Seed");
    boolean isNewFile = fs.createNewFile(outFile);

    List<Text> chosenTexts = new ArrayList<>(k);
    List<Vector> centers = new ArrayList<>(k);

    if (isNewFile) {
      Path inputPathPattern;

      if (fs.getFileStatus(input).isDirectory()) {
        inputPathPattern = new Path(input, "*");
      } else {
        inputPathPattern = input;
      }

      FileStatus[] inputFiles = fs.globStatus(inputPathPattern, PathFilters.logsCRCFilter());

      Random random = (seed != null) ? RandomUtils.getRandom(seed) : RandomUtils.getRandom();

      Pair<String, Vector> c1Data = selectFirstCluster(inputFiles, random, conf);
      chosenTexts.add(new Text(c1Data.getFirst()));
      Vector c1 = c1Data.getSecond();
      centers.add(c1);

      if (k > 1) {
        double regularization = computeRegularization(inputFiles, c1, measure, conf);

        Collection<CandidateEntry> candidateEntries = getPointSamples(inputFiles, k, m, c1, measure, regularization, random, conf);
        Iterator<CandidateEntry> entryIterator = candidateEntries.iterator();
        for (int i=1; i<k; i++) {
          CandidateEntry xEntry = entryIterator.next();
          double dx = distanceToCenters(xEntry.coordinates, centers, measure);
          dx = dx * dx;
          for (int j=1; j<m; j++) {  // we already used one element, the spec says j=2..m
            CandidateEntry yEntry = entryIterator.next();
            double dy = distanceToCenters(xEntry.coordinates, centers, measure);
            dy = dy * dy;

            double r = random.nextDouble();
            double ratio = (dy * xEntry.weight) / (dx * yEntry.weight);
            if (ratio > r) {
              xEntry = yEntry;
            }
          }
          centers.add(xEntry.coordinates);
          chosenTexts.add(new Text(xEntry.text));
        }
      }

      try(SequenceFile.Writer writer =
              SequenceFile.createWriter(fs, conf, outFile, Text.class, ClusterWritable.class)){
        for (int i=0; i<centers.size(); i++) {
          Vector center = centers.get(i);
          Kluster newCluster = new Kluster(center, i, measure);
          newCluster.observe(center, 1);
          writer.append(chosenTexts.get(i), new ClusterWritable(newCluster));
        }
      }
    }

    return outFile;
  }

  /*
   * Selected first cluster using https://en.wikipedia.org/wiki/Reservoir_sampling
   *
   * TODO: Map-Reduce
   */
  private static Pair<String, Vector> selectFirstCluster(FileStatus[] inputFiles, Random random, Configuration conf) {
    int n = 0;
    String c1Repr = null;
    Vector c1 = null;

    for (FileStatus fileStatus : inputFiles) {
      if (!fileStatus.isDirectory()) {
        for (Pair<Writable, VectorWritable> record: new SequenceFileIterable<Writable, VectorWritable>(fileStatus.getPath(), true, conf)) {
          if (n == 0 || random.nextInt(n) == 0) {  // the reservoir size = 1, so the chance should be 1/n, the chance of 0
            c1Repr = record.getFirst().toString();
            c1 = record.getSecond().get();
          }
          n++;
        }
      }
    }

    if (n == 0) {
      throw new IllegalStateException("In order to perform clustering there should be some samples");
    }

    return new Pair<>(c1Repr, c1);
  }

  /*
   * Computes the regularization factor in q(x|c), namely Sum_{x}(d(x, c))/n
   *
   * TODO: use map-reduce
   * TODO: maybe save the distances in order not to compute them twice (here and when selecting candidates)
   */
  private static double computeRegularization(FileStatus[] inputFiles, Vector c, DistanceMeasure measure, Configuration conf) {
    int n = 0;
    double totalSquaredDistance = 0;
    for (FileStatus fileStatus : inputFiles) {
      if (!fileStatus.isDirectory()) {
        for (Pair<Writable, VectorWritable> record : new SequenceFileIterable<Writable, VectorWritable>(fileStatus.getPath(), true, conf)) {
          n++;
          double d = measure.distance(record.getSecond().get(), c);
          totalSquaredDistance += d * d;
        }
      }
    }
    return totalSquaredDistance / n;
  }

  /*
   * Generate samples for the main loop of Assumption-Free K-MC2 algorithm ( https://papers.nips.cc/paper/6478-fast-and-provably-good-seedings-for-k-means.pdf )
   *
   * Collects cluster `(k-1)*m` candidates using Weighted Random Sampling using Reservoir.
   * As described in https://en.wikipedia.org/wiki/Reservoir_sampling#Algorithm_A-Res
   *
   * Note that the original paper the sampling is done with replacement, but without replacement is good enough for non-small datasets.
   */
  private static Collection<CandidateEntry> getPointSamples(FileStatus[] inputFiles, int k, int m, Vector c1, DistanceMeasure measure, double regularization, Random random, Configuration conf) {
    int candidatesCount = (k - 1) * m;
    PriorityQueue<CandidateEntry> queue = new PriorityQueue<>(candidatesCount);

    // This should be done in map-reduce
    for (FileStatus fileStatus : inputFiles) {
      if (!fileStatus.isDirectory()) {
        for (Pair<Writable, VectorWritable> record : new SequenceFileIterable<Writable, VectorWritable>(fileStatus.getPath(), true, conf)) {
          Vector candidate = record.getSecond().get();
          double d = measure.distance(candidate, c1); // NOTE: unfortunately this is a bit of waste of resources due to abstraction since the distance is sqrt-ed and then squared again
          double weight = d * d + regularization;  // q(x|c1) * Sum_y(d(y, c1)) * 2
          double rand = random.nextDouble();  // FIXME: the interval should be 0.0-1.0 (inclusive), but the java API excludes 1.0
          double priority = Math.pow(rand, 1 / weight);  // weight is not 0 thanks to the regularization
          if (queue.size() < candidatesCount) {
            queue.add(new CandidateEntry(priority, candidate, weight, record.getFirst().toString()));
          } else if (queue.peek().priority < priority) {
            queue.poll();
            queue.add(new CandidateEntry(priority, candidate, weight, record.getFirst().toString()));
          }
        }
      }
    }
    // Hopefully the order is not that important
    if (queue.size() == candidatesCount) {
      return queue;
    } else {  // this should not happen and if it does why is someone clustering so few elements with something other than random
      ArrayList<CandidateEntry> result = new ArrayList<>(candidatesCount);
      Preconditions.checkArgument(queue.size() > 0, "Unexpected empty queue found - probably there are not elements to cluster");
      while (result.size() < candidatesCount) {
        for (CandidateEntry c : queue) {
          result.add(c);
          if (result.size() == candidatesCount)
            break;
        }
      }
      return result;
    }
  }

  /*
   * Compute the minimum distance to any of the centers
   */
  private static double distanceToCenters(Vector x, List<Vector> centers, DistanceMeasure measure) {
    double min_distance = 0;
    for (Vector c : centers) {
      double d = measure.distance(x, c);
      if (d < min_distance) {
        min_distance = d;
      }
    }
    return min_distance;
  }

}
