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
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.vectors.VectorHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ClusterDumper {

  private static final Logger log = LoggerFactory.getLogger(ClusterDumper.class);

  private final Path seqFileDir;

  private final Path pointsDir;

  private String termDictionary;

  private String dictionaryFormat;

  private String outputFile;

  private int subString = Integer.MAX_VALUE;

  private int numTopFeatures = 10;

  private Map<Integer, List<WeightedVectorWritable>> clusterIdToPoints = null;

  private boolean useJSON = false;

  public ClusterDumper(Path seqFileDir, Path pointsDir) throws IOException {
    this.seqFileDir = seqFileDir;
    this.pointsDir = pointsDir;
    init();
  }

  private void init() throws IOException {
    if (this.pointsDir != null) {
      JobConf conf = new JobConf(Job.class);
      // read in the points
      clusterIdToPoints = readPoints(this.pointsDir, conf);
    } else {
      clusterIdToPoints = Collections.emptyMap();
    }
  }

  public void printClusters(String[] dictionary) throws IOException, InstantiationException, IllegalAccessException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(Job.class);
    client.setConf(conf);

    if (this.termDictionary != null) {
      if (dictionaryFormat.equals("text")) {
        dictionary = VectorHelper.loadTermDictionary(new File(this.termDictionary));
      } else if (dictionaryFormat.equals("sequencefile")) {
        FileSystem fs = FileSystem.get(new Path(this.termDictionary).toUri(), conf);
        dictionary = VectorHelper.loadTermDictionary(conf, fs, this.termDictionary);
      } else {
        throw new IllegalArgumentException("Invalid dictionary format");
      }
    }

    Writer writer = this.outputFile == null ? new OutputStreamWriter(System.out) : new FileWriter(this.outputFile);

    FileSystem fs = seqFileDir.getFileSystem(conf);
    FileStatus[] seqFileList = fs.listStatus(seqFileDir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        return !path.getName().endsWith(".crc");
      }
    });
    for (FileStatus seqFile : seqFileList) {
      Path path = seqFile.getPath();
      System.out.println("Input Path: " + path);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      Writable key = (Writable) reader.getKeyClass().newInstance();
      Writable value = (Writable) reader.getValueClass().newInstance();
      while (reader.next(key, value)) {
        Cluster cluster = (Cluster) value;
        String fmtStr = useJSON ? cluster.asJsonString() : cluster.asFormatString(dictionary);
        if (subString > 0 && fmtStr.length() > subString) {
          writer.append(":").append(fmtStr.substring(0, Math.min(subString, fmtStr.length())));
        } else {
          writer.append(fmtStr);
        }

        writer.append('\n');

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
            writer.append(Double.toString(point.getWeight())).append(": ");
            writer.append(ClusterBase.formatVector(point.getVector().get(), dictionary));
            if (iterator.hasNext()) {
              writer.append("\n\t");
            }
          }
          writer.write('\n');
        }
        writer.flush();
      }
      reader.close();
    }
    if (this.outputFile != null) {
      writer.flush();
      writer.close();
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

  public static void main(String[] args) throws IOException, IllegalAccessException, InstantiationException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option seqOpt = obuilder.withLongName("seqFileDir").withRequired(false).withArgument(
        abuilder.withName("seqFileDir").withMinimum(1).withMaximum(1).create()).withDescription(
        "The directory containing Sequence Files for the Clusters").withShortName("s").create();
    Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
        abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
        "The output file.  If not specified, dumps to the console").withShortName("o").create();
    Option substringOpt = obuilder.withLongName("substring").withRequired(false).withArgument(
        abuilder.withName("substring").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of chars of the asFormatString() to print").withShortName("b").create();
    Option numWordsOpt = obuilder.withLongName("numWords").withRequired(false).withArgument(
        abuilder.withName("numWords").withMinimum(1).withMaximum(1).create()).withDescription("The number of top terms to print")
        .withShortName("n").create();
    Option centroidJSonOpt = obuilder.withLongName("json").withRequired(false).withDescription(
        "Output the centroid as JSON.  Otherwise it substitues in the terms for vector cell entries").withShortName("j").create();
    Option pointsOpt = obuilder.withLongName("pointsDir").withRequired(false).withArgument(
        abuilder.withName("pointsDir").withMinimum(1).withMaximum(1).create()).withDescription(
        "The directory containing points sequence files mapping input vectors to their cluster.  "
            + "If specified, then the program will output the points associated with a cluster").withShortName("p").create();
    Option dictOpt = obuilder.withLongName("dictionary").withRequired(false).withArgument(
        abuilder.withName("dictionary").withMinimum(1).withMaximum(1).create()).withDescription("The dictionary file. ")
        .withShortName("d").create();
    Option dictTypeOpt = obuilder.withLongName("dictionaryType").withRequired(false).withArgument(
        abuilder.withName("dictionaryType").withMinimum(1).withMaximum(1).create()).withDescription(
        "The dictionary file type (text|sequencefile)").withShortName("dt").create();
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(helpOpt).withOption(seqOpt).withOption(outputOpt)
        .withOption(substringOpt).withOption(pointsOpt).withOption(centroidJSonOpt).withOption(dictOpt).withOption(dictTypeOpt)
        .withOption(numWordsOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      if (!cmdLine.hasOption(seqOpt)) {
        return;
      }
      Path seqFileDir = new Path(cmdLine.getValue(seqOpt).toString());
      String termDictionary = null;
      if (cmdLine.hasOption(dictOpt)) {
        termDictionary = cmdLine.getValue(dictOpt).toString();
      }

      Path pointsDir = null;
      if (cmdLine.hasOption(pointsOpt)) {
        pointsDir = new Path(cmdLine.getValue(pointsOpt).toString());
      }
      String outputFile = null;
      if (cmdLine.hasOption(outputOpt)) {
        outputFile = cmdLine.getValue(outputOpt).toString();
      }

      int sub = -1;
      if (cmdLine.hasOption(substringOpt)) {
        sub = Integer.parseInt(cmdLine.getValue(substringOpt).toString());
      }

      ClusterDumper clusterDumper = new ClusterDumper(seqFileDir, pointsDir);
      if (cmdLine.hasOption(centroidJSonOpt)) {
        clusterDumper.setUseJSON(true);
      }

      if (outputFile != null) {
        clusterDumper.setOutputFile(outputFile);
      }

      String dictionaryType = "text";
      if (cmdLine.hasOption(dictTypeOpt)) {
        dictionaryType = cmdLine.getValue(dictTypeOpt).toString();
      }

      if (termDictionary != null) {
        clusterDumper.setTermDictionary(termDictionary, dictionaryType);
      }

      if (cmdLine.hasOption(numWordsOpt)) {
        int numWords = Integer.parseInt(cmdLine.getValue(numWordsOpt).toString());
        clusterDumper.setNumTopFeatures(numWords);
      }

      if (sub >= 0) {
        clusterDumper.setSubString(sub);
      }
      clusterDumper.printClusters(null);
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }

  private void setUseJSON(boolean json) {
    this.useJSON = json;
  }

  private static Map<Integer, List<WeightedVectorWritable>> readPoints(Path pointsPathDir, JobConf conf)
      throws IOException {
    Map<Integer, List<WeightedVectorWritable>> result = new TreeMap<Integer, List<WeightedVectorWritable>>();

    FileSystem fs = pointsPathDir.getFileSystem(conf);
    FileStatus[] children = fs.listStatus(pointsPathDir, new PathFilter() {
      @Override
      public boolean accept(Path path) {
        return !path.getName().endsWith(".crc");
      }
    });

    for (FileStatus file : children) {
      Path path = file.getPath();
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      try {
        IntWritable key = (IntWritable) reader.getKeyClass().newInstance();
        WeightedVectorWritable value = (WeightedVectorWritable) reader.getValueClass().newInstance();
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
          value = (WeightedVectorWritable) reader.getValueClass().newInstance();
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

    private double weight = 0.0;

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

    List<Pair<String, Double>> topTerms = new LinkedList<Pair<String, Double>>();

    for (int i = 0; (i < vectorTerms.size()) && (i < numTerms); i++) {
      int index = vectorTerms.get(i).index;
      String dictTerm = dictionary[index];
      if (dictTerm == null) {
        log.error("Dictionary entry missing for {}", index);
        continue;
      }
      topTerms.add(new Pair<String, Double>(dictTerm, vectorTerms.get(i).weight));
    }

    StringBuilder sb = new StringBuilder();

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
