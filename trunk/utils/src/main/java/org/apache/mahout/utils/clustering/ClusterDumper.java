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
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.vectors.VectorHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ClusterDumper extends AbstractJob {

  public static final String OUTPUT_OPTION = "output";

  public static final String DICTIONARY_TYPE_OPTION = "dictionaryType";

  public static final String DICTIONARY_OPTION = "dictionary";

  public static final String POINTS_DIR_OPTION = "pointsDir";

  public static final String JSON_OPTION = "json";

  public static final String NUM_WORDS_OPTION = "numWords";

  public static final String SUBSTRING_OPTION = "substring";

  public static final String SEQ_FILE_DIR_OPTION = "seqFileDir";

  private static final Logger log = LoggerFactory.getLogger(ClusterDumper.class);

  private Path seqFileDir;

  private Path pointsDir;

  private String termDictionary;

  private String dictionaryFormat;

  private String outputFile;

  private int subString = Integer.MAX_VALUE;

  private int numTopFeatures = 10;

  private Map<Integer, List<WeightedVectorWritable>> clusterIdToPoints;

  private boolean useJSON;

  public ClusterDumper(Path seqFileDir, Path pointsDir) throws IOException {
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
    addOption(SEQ_FILE_DIR_OPTION, "s", "The directory containing Sequence Files for the Clusters", true);
    addOption(OUTPUT_OPTION, "o", "Optional output directory. Default is to output to the console.");
    addOption(SUBSTRING_OPTION, "b", "The number of chars of the asFormatString() to print");
    addOption(NUM_WORDS_OPTION, "n", "The number of top terms to print");
    addOption(JSON_OPTION, "j",
        "Output the centroid as JSON.  Otherwise it substitues in the terms for vector cell entries");
    addOption(POINTS_DIR_OPTION, "p",
        "The directory containing points sequence files mapping input vectors to their cluster.  "
            + "If specified, then the program will output the points associated with a cluster");
    addOption(DICTIONARY_OPTION, "d", "The dictionary file");
    addOption(DICTIONARY_TYPE_OPTION, "dt", "The dictionary file type (text|sequencefile)", "text");
    if (parseArguments(args) == null) {
      return -1;
    }

    seqFileDir = new Path(getOption(SEQ_FILE_DIR_OPTION));
    if (hasOption(POINTS_DIR_OPTION)) {
      pointsDir = new Path(getOption(POINTS_DIR_OPTION));
    }
    outputFile = getOption(OUTPUT_OPTION);
    if (hasOption(SUBSTRING_OPTION)) {
      int sub = Integer.parseInt(getOption(SUBSTRING_OPTION));
      if (sub >= 0) {
        subString = sub;
      }
    }
    if (hasOption(JSON_OPTION)) {
      useJSON = true;
    }
    termDictionary = getOption(DICTIONARY_OPTION);
    dictionaryFormat = getOption(DICTIONARY_TYPE_OPTION);
    if (hasOption(NUM_WORDS_OPTION)) {
      numTopFeatures = Integer.parseInt(getOption(NUM_WORDS_OPTION));
    }
    init();
    printClusters(null);
    return 0;
  }

  public void printClusters(String[] dictionary) throws IOException, InstantiationException, IllegalAccessException {
    Configuration conf = new Configuration();

    if (this.termDictionary != null) {
      if ("text".equals(dictionaryFormat)) {
        dictionary = VectorHelper.loadTermDictionary(new File(this.termDictionary));
      } else if ("sequencefile".equals(dictionaryFormat)) {
        FileSystem fs = FileSystem.get(new Path(this.termDictionary).toUri(), conf);
        dictionary = VectorHelper.loadTermDictionary(conf, fs, this.termDictionary);
      } else {
        throw new IllegalArgumentException("Invalid dictionary format");
      }
    }

    Writer writer = this.outputFile == null ? new OutputStreamWriter(System.out) : new FileWriter(this.outputFile);
    try {
      FileSystem fs = seqFileDir.getFileSystem(conf);
      for (FileStatus seqFile : fs.globStatus(new Path(seqFileDir, "part-*"))) {
        Path path = seqFile.getPath();
        //System.out.println("Input Path: " + path); doesn't this interfere with output?
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
        try {
          Writable key = reader.getKeyClass().asSubclass(Writable.class).newInstance();
          Writable value = reader.getValueClass().asSubclass(Writable.class).newInstance();
          while (reader.next(key, value)) {
            Cluster cluster = (Cluster) value;
            String fmtStr = useJSON ? cluster.asJsonString() : cluster.asFormatString(dictionary);
            if (subString > 0 && fmtStr.length() > subString) {
              writer.write(':');
              writer.write(fmtStr, 0, Math.min(subString, fmtStr.length()));
            } else {
              writer.write(fmtStr);
            }

            writer.write('\n');

            if (dictionary != null) {
              String topTerms = getTopFeatures(cluster.getCenter(), dictionary, numTopFeatures);
              writer.write("\tTop Terms: ");
              writer.write(topTerms);
              writer.write('\n');
            }

            List<WeightedVectorWritable> points = clusterIdToPoints.get(cluster.getId());
            if (points != null) {
              writer.write("\tWeight:  Point:\n\t");
              for (Iterator<WeightedVectorWritable> iterator = points.iterator(); iterator.hasNext();) {
                WeightedVectorWritable point = iterator.next();
                writer.write(String.valueOf(point.getWeight()));
                writer.write(": ");
                writer.write(AbstractCluster.formatVector(point.getVector(), dictionary));
                if (iterator.hasNext()) {
                  writer.write("\n\t");
                }
              }
              writer.write('\n');
            }
          }
        } finally {
          reader.close();
        }
      }
    } finally {
      writer.close();
    }
  }

  private void init() throws IOException {
    if (this.pointsDir != null) {
      Configuration conf = new Configuration();
      // read in the points
      clusterIdToPoints = readPoints(this.pointsDir, conf);
    } else {
      clusterIdToPoints = Collections.emptyMap();
    }
  }

  public String getOutputFile() {
    return outputFile;
  }

  public void setOutputFile(String outputFile) {
    this.outputFile = outputFile;
  }

  public int getSubString() {
    return subString;
  }

  public void setSubString(int subString) {
    this.subString = subString;
  }

  public Map<Integer, List<WeightedVectorWritable>> getClusterIdToPoints() {
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

  private static Map<Integer, List<WeightedVectorWritable>> readPoints(Path pointsPathDir,
                                                                       Configuration conf) throws IOException {
    Map<Integer, List<WeightedVectorWritable>> result = new TreeMap<Integer, List<WeightedVectorWritable>>();

    FileSystem fs = pointsPathDir.getFileSystem(conf);
    FileStatus[] children = fs.listStatus(pointsPathDir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        String name = path.getName();
        return !(name.endsWith(".crc") || name.startsWith("_"));
      }
    });

    for (FileStatus file : children) {
      Path path = file.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      try {
        IntWritable key = reader.getKeyClass().asSubclass(IntWritable.class).newInstance();
        WeightedVectorWritable value = reader.getValueClass().asSubclass(WeightedVectorWritable.class).newInstance();
        while (reader.next(key, value)) {
          // value is the cluster id as an int, key is the name/id of the
          // vector, but that doesn't matter because we only care about printing
          // it
          //String clusterId = value.toString();
          List<WeightedVectorWritable> pointList = result.get(key.get());
          if (pointList == null) {
            pointList = new ArrayList<WeightedVectorWritable>();
            result.put(key.get(), pointList);
          }
          pointList.add(value);
          value = reader.getValueClass().asSubclass(WeightedVectorWritable.class).newInstance();
        }
      } catch (InstantiationException e) {
        log.error("Exception", e);
      } catch (IllegalAccessException e) {
        log.error("Exception", e);
      }
    }

    return result;
  }

  static class TermIndexWeight {
    private int index = -1;

    private final double weight;

    TermIndexWeight(int index, double weight) {
      this.index = index;
      this.weight = weight;
    }
  }

  private static String getTopFeatures(Vector vector, String[] dictionary, int numTerms) {

    List<TermIndexWeight> vectorTerms = new ArrayList<TermIndexWeight>();

    Iterator<Vector.Element> iter = vector.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element elt = iter.next();
      vectorTerms.add(new TermIndexWeight(elt.index(), elt.get()));
    }

    // Sort results in reverse order (ie weight in descending order)
    Collections.sort(vectorTerms, new Comparator<TermIndexWeight>() {
      @Override
      public int compare(TermIndexWeight one, TermIndexWeight two) {
        return Double.compare(two.weight, one.weight);
      }
    });

    Collection<Pair<String, Double>> topTerms = new LinkedList<Pair<String, Double>>();

    for (int i = 0; (i < vectorTerms.size()) && (i < numTerms); i++) {
      int index = vectorTerms.get(i).index;
      String dictTerm = dictionary[index];
      if (dictTerm == null) {
        log.error("Dictionary entry missing for {}", index);
        continue;
      }
      topTerms.add(new Pair<String, Double>(dictTerm, vectorTerms.get(i).weight));
    }

    StringBuilder sb = new StringBuilder(100);

    for (Pair<String, Double> item : topTerms) {
      String term = item.getFirst();
      sb.append("\n\t\t");
      sb.append(StringUtils.rightPad(term, 40));
      sb.append("=>");
      sb.append(StringUtils.leftPad(item.getSecond().toString(), 20));
    }
    return sb.toString();
  }

}
