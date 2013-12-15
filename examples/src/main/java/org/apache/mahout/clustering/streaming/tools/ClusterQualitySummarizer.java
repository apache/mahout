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

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.ClusteringUtils;
import org.apache.mahout.clustering.streaming.mapreduce.CentroidWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.stats.OnlineSummarizer;

public class ClusterQualitySummarizer extends AbstractJob {
  private String outputFile;

  private PrintWriter fileOut;

  private String trainFile;
  private String testFile;
  private String centroidFile;
  private String centroidCompareFile;
  private boolean mahoutKMeansFormat;
  private boolean mahoutKMeansFormatCompare;

  private DistanceMeasure distanceMeasure = new SquaredEuclideanDistanceMeasure();

  public void printSummaries(List<OnlineSummarizer> summarizers, String type) {
    printSummaries(summarizers, type, fileOut);
  }

  public static void printSummaries(List<OnlineSummarizer> summarizers, String type, PrintWriter fileOut) {
    double maxDistance = 0;
    for (int i = 0; i < summarizers.size(); ++i) {
      OnlineSummarizer summarizer = summarizers.get(i);
      if (summarizer.getCount() > 1) {
        maxDistance = Math.max(maxDistance, summarizer.getMax());
        System.out.printf("Average distance in cluster %d [%d]: %f\n", i, summarizer.getCount(), summarizer.getMean());
        // If there is just one point in the cluster, quartiles cannot be estimated. We'll just assume all the quartiles
        // equal the only value.
        if (fileOut != null) {
          fileOut.printf("%d,%f,%f,%f,%f,%f,%f,%f,%d,%s\n", i, summarizer.getMean(),
              summarizer.getSD(),
              summarizer.getQuartile(0),
              summarizer.getQuartile(1),
              summarizer.getQuartile(2),
              summarizer.getQuartile(3),
              summarizer.getQuartile(4), summarizer.getCount(), type);
        }
      } else {
        System.out.printf("Cluster %d is has %d data point. Need atleast 2 data points in a cluster for" +
            " OnlineSummarizer.\n", i, summarizer.getCount());
      }
    }
    System.out.printf("Num clusters: %d; maxDistance: %f\n", summarizers.size(), maxDistance);
  }

  public int run(String[] args) throws IOException {
    if (!parseArgs(args)) {
      return -1;
    }

    Configuration conf = new Configuration();
    try {
//      Configuration.dumpConfiguration(conf, new OutputStreamWriter(System.out));

      fileOut = new PrintWriter(new FileOutputStream(outputFile));
      fileOut.printf("cluster,distance.mean,distance.sd,distance.q0,distance.q1,distance.q2,distance.q3,"
          + "distance.q4,count,is.train\n");

      // Reading in the centroids (both pairs, if they exist).
      List<Centroid> centroids;
      List<Centroid> centroidsCompare = null;
      if (mahoutKMeansFormat) {
        SequenceFileDirValueIterable<ClusterWritable> clusterIterable =
            new SequenceFileDirValueIterable<ClusterWritable>(new Path(centroidFile), PathType.GLOB, conf);
        centroids = Lists.newArrayList(IOUtils.getCentroidsFromClusterWritableIterable(clusterIterable));
      } else {
        SequenceFileDirValueIterable<CentroidWritable> centroidIterable =
            new SequenceFileDirValueIterable<CentroidWritable>(new Path(centroidFile), PathType.GLOB, conf);
        centroids = Lists.newArrayList(IOUtils.getCentroidsFromCentroidWritableIterable(centroidIterable));
      }

      if (centroidCompareFile != null) {
        if (mahoutKMeansFormatCompare) {
          SequenceFileDirValueIterable<ClusterWritable> clusterCompareIterable =
              new SequenceFileDirValueIterable<ClusterWritable>(new Path(centroidCompareFile), PathType.GLOB, conf);
          centroidsCompare = Lists.newArrayList(
              IOUtils.getCentroidsFromClusterWritableIterable(clusterCompareIterable));
        } else {
          SequenceFileDirValueIterable<CentroidWritable> centroidCompareIterable =
              new SequenceFileDirValueIterable<CentroidWritable>(new Path(centroidCompareFile), PathType.GLOB, conf);
          centroidsCompare = Lists.newArrayList(
              IOUtils.getCentroidsFromCentroidWritableIterable(centroidCompareIterable));
        }
      }

      // Reading in the "training" set.
      SequenceFileDirValueIterable<VectorWritable> trainIterable =
          new SequenceFileDirValueIterable<VectorWritable>(new Path(trainFile), PathType.GLOB, conf);
      Iterable<Vector> trainDatapoints = IOUtils.getVectorsFromVectorWritableIterable(trainIterable);
      Iterable<Vector> datapoints = trainDatapoints;

      printSummaries(ClusteringUtils.summarizeClusterDistances(trainDatapoints, centroids,
          new SquaredEuclideanDistanceMeasure()), "train");

      // Also adding in the "test" set.
      if (testFile != null) {
        SequenceFileDirValueIterable<VectorWritable> testIterable =
            new SequenceFileDirValueIterable<VectorWritable>(new Path(testFile), PathType.GLOB, conf);
        Iterable<Vector> testDatapoints = IOUtils.getVectorsFromVectorWritableIterable(testIterable);

        printSummaries(ClusteringUtils.summarizeClusterDistances(testDatapoints, centroids,
            new SquaredEuclideanDistanceMeasure()), "test");

        datapoints = Iterables.concat(trainDatapoints, testDatapoints);
      }

      // At this point, all train/test CSVs have been written. We now compute quality metrics.
      List<OnlineSummarizer> summaries =
          ClusteringUtils.summarizeClusterDistances(datapoints, centroids, distanceMeasure);
      List<OnlineSummarizer> compareSummaries = null;
      if (centroidsCompare != null) {
        compareSummaries = ClusteringUtils.summarizeClusterDistances(datapoints, centroidsCompare, distanceMeasure);
      }
      System.out.printf("[Dunn Index] First: %f", ClusteringUtils.dunnIndex(centroids, distanceMeasure, summaries));
      if (compareSummaries != null) {
        System.out.printf(" Second: %f\n", ClusteringUtils.dunnIndex(centroidsCompare, distanceMeasure, compareSummaries));
      } else {
        System.out.printf("\n");
      }
      System.out.printf("[Davies-Bouldin Index] First: %f",
          ClusteringUtils.daviesBouldinIndex(centroids, distanceMeasure, summaries));
      if (compareSummaries != null) {
        System.out.printf(" Second: %f\n",
          ClusteringUtils.daviesBouldinIndex(centroidsCompare, distanceMeasure, compareSummaries));
      } else {
        System.out.printf("\n");
      }
    } catch (IOException e) {
      System.out.println(e.getMessage());
    } finally {
      Closeables.close(fileOut, false);
    }
    return 0;
  }

  private boolean parseArgs(String[] args) {
    DefaultOptionBuilder builder = new DefaultOptionBuilder();

    Option help = builder.withLongName("help").withDescription("print this list").create();

    ArgumentBuilder argumentBuilder = new ArgumentBuilder();
    Option inputFileOption = builder.withLongName("input")
        .withShortName("i")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("input").withMaximum(1).create())
        .withDescription("where to get seq files with the vectors (training set)")
        .create();

    Option testInputFileOption = builder.withLongName("testInput")
        .withShortName("itest")
        .withArgument(argumentBuilder.withName("testInput").withMaximum(1).create())
        .withDescription("where to get seq files with the vectors (test set)")
        .create();

    Option centroidsFileOption = builder.withLongName("centroids")
        .withShortName("c")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("centroids").withMaximum(1).create())
        .withDescription("where to get seq files with the centroids (from Mahout KMeans or StreamingKMeansDriver)")
        .create();

    Option centroidsCompareFileOption = builder.withLongName("centroidsCompare")
        .withShortName("cc")
        .withRequired(false)
        .withArgument(argumentBuilder.withName("centroidsCompare").withMaximum(1).create())
        .withDescription("where to get seq files with the second set of centroids (from Mahout KMeans or "
            + "StreamingKMeansDriver)")
        .create();

    Option outputFileOption = builder.withLongName("output")
        .withShortName("o")
        .withRequired(true)
        .withArgument(argumentBuilder.withName("output").withMaximum(1).create())
        .withDescription("where to dump the CSV file with the results")
        .create();

    Option mahoutKMeansFormatOption = builder.withLongName("mahoutkmeansformat")
        .withShortName("mkm")
        .withDescription("if set, read files as (IntWritable, ClusterWritable) pairs")
        .withArgument(argumentBuilder.withName("numpoints").withMaximum(1).create())
        .create();

    Option mahoutKMeansCompareFormatOption = builder.withLongName("mahoutkmeansformatCompare")
        .withShortName("mkmc")
        .withDescription("if set, read files as (IntWritable, ClusterWritable) pairs")
        .withArgument(argumentBuilder.withName("numpoints").withMaximum(1).create())
        .create();

    Group normalArgs = new GroupBuilder()
        .withOption(help)
        .withOption(inputFileOption)
        .withOption(testInputFileOption)
        .withOption(outputFileOption)
        .withOption(centroidsFileOption)
        .withOption(centroidsCompareFileOption)
        .withOption(mahoutKMeansFormatOption)
        .withOption(mahoutKMeansCompareFormatOption)
        .create();

    Parser parser = new Parser();
    parser.setHelpOption(help);
    parser.setHelpTrigger("--help");
    parser.setGroup(normalArgs);
    parser.setHelpFormatter(new HelpFormatter(" ", "", " ", 150));

    CommandLine cmdLine = parser.parseAndHelp(args);
    if (cmdLine == null) {
      return false;
    }

    trainFile = (String) cmdLine.getValue(inputFileOption);
    if (cmdLine.hasOption(testInputFileOption)) {
      testFile = (String) cmdLine.getValue(testInputFileOption);
    }
    centroidFile = (String) cmdLine.getValue(centroidsFileOption);
    if (cmdLine.hasOption(centroidsCompareFileOption)) {
      centroidCompareFile = (String) cmdLine.getValue(centroidsCompareFileOption);
    }
    outputFile = (String) cmdLine.getValue(outputFileOption);
    if (cmdLine.hasOption(mahoutKMeansFormatOption)) {
      mahoutKMeansFormat = true;
    }
    if (cmdLine.hasOption(mahoutKMeansCompareFormatOption)) {
      mahoutKMeansFormatCompare = true;
    }
    return true;
  }

  public static void main(String[] args) throws IOException {
    new ClusterQualitySummarizer().run(args);
  }
}
