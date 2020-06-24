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

package org.apache.mahout.utils.clustering;

import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.commons.io.Charsets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.clustering.cdbw.CDbwEvaluator;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.evaluation.ClusterEvaluator;
import org.apache.mahout.clustering.evaluation.RepresentativePointsDriver;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.utils.vectors.VectorHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ClusterDumper extends AbstractJob {

  public static final String SAMPLE_POINTS = "samplePoints";
  DistanceMeasure measure;

  public enum OUTPUT_FORMAT {
    TEXT,
    CSV,
    GRAPH_ML,
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1343
    JSON,
  }

  public static final String DICTIONARY_TYPE_OPTION = "dictionaryType";
  public static final String DICTIONARY_OPTION = "dictionary";
  public static final String POINTS_DIR_OPTION = "pointsDir";
  public static final String NUM_WORDS_OPTION = "numWords";
  public static final String SUBSTRING_OPTION = "substring";
  public static final String EVALUATE_CLUSTERS = "evaluate";

  public static final String OUTPUT_FORMAT_OPT = "outputFormat";

  private static final Logger log = LoggerFactory.getLogger(ClusterDumper.class);
  private Path seqFileDir;
  private Path pointsDir;
  private long maxPointsPerCluster = Long.MAX_VALUE;
  private String termDictionary;
  private String dictionaryFormat;
  private int subString = Integer.MAX_VALUE;
  private int numTopFeatures = 10;
  private Map<Integer, List<WeightedPropertyVectorWritable>> clusterIdToPoints;
  private OUTPUT_FORMAT outputFormat = OUTPUT_FORMAT.TEXT;
  private boolean runEvaluation;

  public ClusterDumper(Path seqFileDir, Path pointsDir) {
    this.seqFileDir = seqFileDir;
    this.pointsDir = pointsDir;
    init();
  }

  public ClusterDumper() {
    setConf(new Configuration());
  }

  public static void main(String[] args) throws Exception {
    new ClusterDumper().run(args);
  }

  @Override
  public int run(String[] args) throws Exception {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-947
    addInputOption();
    addOutputOption();
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1343
    addOption(OUTPUT_FORMAT_OPT, "of", "The optional output format for the results.  Options: TEXT, CSV, JSON or GRAPH_ML",
        "TEXT");
    addOption(SUBSTRING_OPTION, "b", "The number of chars of the asFormatString() to print");
    addOption(NUM_WORDS_OPTION, "n", "The number of top terms to print");
    addOption(POINTS_DIR_OPTION, "p",
//IC see: https://issues.apache.org/jira/browse/MAHOUT-761
            "The directory containing points sequence files mapping input vectors to their cluster.  "
                    + "If specified, then the program will output the points associated with a cluster");
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1173
    addOption(SAMPLE_POINTS, "sp", "Specifies the maximum number of points to include _per_ cluster.  The default "
        + "is to include all points");
    addOption(DICTIONARY_OPTION, "d", "The dictionary file");
    addOption(DICTIONARY_TYPE_OPTION, "dt", "The dictionary file type (text|sequencefile)", "text");
    addOption(buildOption(EVALUATE_CLUSTERS, "e", "Run ClusterEvaluator and CDbwEvaluator over the input.  "
        + "The output will be appended to the rest of the output at the end.", false, false, null));
    addOption(DefaultOptionCreator.distanceMeasureOption().create());

    // output is optional, will print to System.out per default
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1075
    if (parseArguments(args, false, true) == null) {
      return -1;
    }

//IC see: https://issues.apache.org/jira/browse/MAHOUT-947
    seqFileDir = getInputPath();
    if (hasOption(POINTS_DIR_OPTION)) {
      pointsDir = new Path(getOption(POINTS_DIR_OPTION));
    }
    outputFile = getOutputFile();
    if (hasOption(SUBSTRING_OPTION)) {
      int sub = Integer.parseInt(getOption(SUBSTRING_OPTION));
//IC see: https://issues.apache.org/jira/browse/MAHOUT-289
      if (sub >= 0) {
        subString = sub;
      }
    }
    termDictionary = getOption(DICTIONARY_OPTION);
    dictionaryFormat = getOption(DICTIONARY_TYPE_OPTION);
    if (hasOption(NUM_WORDS_OPTION)) {
      numTopFeatures = Integer.parseInt(getOption(NUM_WORDS_OPTION));
    }
//IC see: https://issues.apache.org/jira/browse/MAHOUT-798
    if (hasOption(OUTPUT_FORMAT_OPT)) {
      outputFormat = OUTPUT_FORMAT.valueOf(getOption(OUTPUT_FORMAT_OPT));
    }
//IC see: https://issues.apache.org/jira/browse/MAHOUT-987
    if (hasOption(SAMPLE_POINTS)) {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-899
      maxPointsPerCluster = Long.parseLong(getOption(SAMPLE_POINTS));
    } else {
      maxPointsPerCluster = Long.MAX_VALUE;
    }
//IC see: https://issues.apache.org/jira/browse/MAHOUT-867
    runEvaluation = hasOption(EVALUATE_CLUSTERS);
    String distanceMeasureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    measure = ClassUtils.instantiateAs(distanceMeasureClass, DistanceMeasure.class);

//IC see: https://issues.apache.org/jira/browse/MAHOUT-167
//IC see: https://issues.apache.org/jira/browse/MAHOUT-427
    init();
    printClusters(null);
    return 0;
  }

  public void printClusters(String[] dictionary) throws Exception {
    Configuration conf = new Configuration();
//IC see: https://issues.apache.org/jira/browse/MAHOUT-167

    if (this.termDictionary != null) {
      if ("text".equals(dictionaryFormat)) {
        dictionary = VectorHelper.loadTermDictionary(new File(this.termDictionary));
      } else if ("sequencefile".equals(dictionaryFormat)) {
        dictionary = VectorHelper.loadTermDictionary(conf, this.termDictionary);
      } else {
        throw new IllegalArgumentException("Invalid dictionary format");
      }
    }

    Writer writer;
    boolean shouldClose;
    if (this.outputFile == null) {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-679
      shouldClose = false;
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1184
      writer = new OutputStreamWriter(System.out, Charsets.UTF_8);
    } else {
      shouldClose = true;
//IC see: https://issues.apache.org/jira/browse/MAHOUT-947
      if (outputFile.getName().startsWith("s3n://")) {
        Path p = outputPath;
        FileSystem fs = FileSystem.get(p.toUri(), conf);
        writer = new OutputStreamWriter(fs.create(p), Charsets.UTF_8);
      } else {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1109
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1173
        Files.createParentDirs(outputFile);
        writer = Files.newWriter(this.outputFile, Charsets.UTF_8);
      }
    }
    ClusterWriter clusterWriter = createClusterWriter(writer, dictionary);
    try {
      long numWritten = clusterWriter.write(new SequenceFileDirValueIterable<ClusterWritable>(new Path(seqFileDir,
          "part-*"), PathType.GLOB, conf));

      writer.flush();
//IC see: https://issues.apache.org/jira/browse/MAHOUT-987
      if (runEvaluation) {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-867
        HadoopUtil.delete(conf, new Path("tmp/representative"));
        int numIters = 5;
        RepresentativePointsDriver.main(new String[]{
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1173
          "--input", seqFileDir.toString(),
          "--output", "tmp/representative",
          "--clusteredPoints", pointsDir.toString(),
          "--distanceMeasure", measure.getClass().getName(),
          "--maxIter", String.valueOf(numIters)
        });
        conf.set(RepresentativePointsDriver.DISTANCE_MEASURE_KEY, measure.getClass().getName());
        conf.set(RepresentativePointsDriver.STATE_IN_KEY, "tmp/representative/representativePoints-" + numIters);
        ClusterEvaluator ce = new ClusterEvaluator(conf, seqFileDir);
        writer.append("\n");
        writer.append("Inter-Cluster Density: ").append(String.valueOf(ce.interClusterDensity())).append("\n");
        writer.append("Intra-Cluster Density: ").append(String.valueOf(ce.intraClusterDensity())).append("\n");
        CDbwEvaluator cdbw = new CDbwEvaluator(conf, seqFileDir);
        writer.append("CDbw Inter-Cluster Density: ").append(String.valueOf(cdbw.interClusterDensity())).append("\n");
        writer.append("CDbw Intra-Cluster Density: ").append(String.valueOf(cdbw.intraClusterDensity())).append("\n");
        writer.append("CDbw Separation: ").append(String.valueOf(cdbw.separation())).append("\n");
        writer.flush();
      }
      log.info("Wrote {} clusters", numWritten);
    } finally {
      if (shouldClose) {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1211
        Closeables.close(clusterWriter, false);
      } else {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-987
        if (clusterWriter instanceof GraphMLClusterWriter) {
          clusterWriter.close();
        }
      }
    }
  }

  ClusterWriter createClusterWriter(Writer writer, String[] dictionary) throws IOException {
    ClusterWriter result;

//IC see: https://issues.apache.org/jira/browse/MAHOUT-987
    switch (outputFormat) {
      case TEXT:
//IC see: https://issues.apache.org/jira/browse/MAHOUT-899
        result = new ClusterDumperWriter(writer, clusterIdToPoints, measure, numTopFeatures, dictionary, subString);
        break;
      case CSV:
        result = new CSVClusterWriter(writer, clusterIdToPoints, measure);
        break;
      case GRAPH_ML:
        result = new GraphMLClusterWriter(writer, clusterIdToPoints, measure, numTopFeatures, dictionary, subString);
        break;
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1343
      case JSON:
        result = new JsonClusterWriter(writer, clusterIdToPoints, measure, numTopFeatures, dictionary);
        break;
      default:
        throw new IllegalStateException("Unknown outputformat: " + outputFormat);
    }
    return result;
  }

  /**
   * Convenience function to set the output format during testing.
   */
  public void setOutputFormat(OUTPUT_FORMAT of) {
    outputFormat = of;
  }

  private void init() {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-294
    if (this.pointsDir != null) {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-167
      Configuration conf = new Configuration();
      // read in the points
//IC see: https://issues.apache.org/jira/browse/MAHOUT-899
      clusterIdToPoints = readPoints(this.pointsDir, maxPointsPerCluster, conf);
    } else {
      clusterIdToPoints = Collections.emptyMap();
    }
  }


  public int getSubString() {
    return subString;
  }

  public void setSubString(int subString) {
    this.subString = subString;
  }

  public Map<Integer, List<WeightedPropertyVectorWritable>> getClusterIdToPoints() {
    return clusterIdToPoints;
  }

  public String getTermDictionary() {
    return termDictionary;
  }

  public void setTermDictionary(String termDictionary, String dictionaryType) {
    this.termDictionary = termDictionary;
    this.dictionaryFormat = dictionaryType;
  }

  public void setNumTopFeatures(int num) {
    this.numTopFeatures = num;
  }

  public int getNumTopFeatures() {
    return this.numTopFeatures;
  }

  public long getMaxPointsPerCluster() {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-899
    return maxPointsPerCluster;
  }

  public void setMaxPointsPerCluster(long maxPointsPerCluster) {
    this.maxPointsPerCluster = maxPointsPerCluster;
  }

  public static Map<Integer, List<WeightedPropertyVectorWritable>> readPoints(Path pointsPathDir,
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
                                                                              long maxPointsPerCluster,
                                                                              Configuration conf) {
    Map<Integer, List<WeightedPropertyVectorWritable>> result = new TreeMap<>();
    for (Pair<IntWritable, WeightedPropertyVectorWritable> record
        : new SequenceFileDirIterable<IntWritable, WeightedPropertyVectorWritable>(pointsPathDir, PathType.LIST,
            PathFilters.logsCRCFilter(), conf)) {
      // value is the cluster id as an int, key is the name/id of the
      // vector, but that doesn't matter because we only care about printing it
      //String clusterId = value.toString();
      int keyValue = record.getFirst().get();
      List<WeightedPropertyVectorWritable> pointList = result.get(keyValue);
      if (pointList == null) {
//IC see: https://issues.apache.org/jira/browse/MAHOUT-1652
        pointList = new ArrayList<>();
        result.put(keyValue, pointList);
      }
//IC see: https://issues.apache.org/jira/browse/MAHOUT-987
      if (pointList.size() < maxPointsPerCluster) {
        pointList.add(record.getSecond());
      }
    }
    return result;
  }
}
