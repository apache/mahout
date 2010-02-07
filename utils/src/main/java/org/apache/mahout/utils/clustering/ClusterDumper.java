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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.mahout.clustering.ClusterBase;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.utils.vectors.VectorHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ClusterDumper {
  
  private static final Logger log = LoggerFactory
      .getLogger(ClusterDumper.class);
  
  private final String seqFileDir;
  private final String pointsDir;
  private String termDictionary;
  private String dictionaryFormat;
  private String outputFile;
  private int subString = Integer.MAX_VALUE;
  private Map<String,List<String>> clusterIdToPoints = null;
  private boolean useJSON = false;
  
  public ClusterDumper(String seqFileDir, String pointsDir) throws IOException {
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
  
  public void printClusters() throws IOException,
                             InstantiationException,
                             IllegalAccessException {
    JobClient client = new JobClient();
    JobConf conf = new JobConf(Job.class);
    client.setConf(conf);
    
    String[] dictionary = null;
    if (this.termDictionary != null) {
      if (dictionaryFormat.equals("text")) {
        dictionary = VectorHelper.loadTermDictionary(new File(
            this.termDictionary));
      } else if (dictionaryFormat.equals("sequencefile")) {
        FileSystem fs = FileSystem.get(new Path(this.termDictionary).toUri(),
          conf);
        dictionary = VectorHelper.loadTermDictionary(conf, fs,
          this.termDictionary);
      } else {
        throw new IllegalArgumentException("Invalid dictionary format");
      }
    }
    
    Writer writer = null;
    if (this.outputFile != null) {
      writer = new FileWriter(this.outputFile);
    } else {
      writer = new OutputStreamWriter(System.out);
    }
    
    File[] seqFileList = new File(this.seqFileDir)
        .listFiles(new FilenameFilter() {
          @Override
          public boolean accept(File file, String name) {
            return name.endsWith(".crc") == false;
          }
        });
    for (File seqFile : seqFileList) {
      if (!seqFile.isFile()) {
        continue;
      }
      Path path = new Path(seqFile.getAbsolutePath());
      System.out.println("Input Path: " + path);
      FileSystem fs = FileSystem.get(path.toUri(), conf);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      Writable key = (Writable) reader.getKeyClass().newInstance();
      ClusterBase value = (ClusterBase) reader.getValueClass().newInstance();
      while (reader.next(key, value)) {
        Vector center = value.getCenter();
        String fmtStr = useJSON ? center.asFormatString() : VectorHelper
            .vectorToString(center, dictionary);
        writer.append("Id: ").append(String.valueOf(value.getId())).append(":")
            .append("name:").append(center.getName()).append(":").append(
              fmtStr.substring(0, Math.min(subString, fmtStr.length())))
            .append('\n');
        
        if (dictionary != null) {
          String topTerms = getTopFeatures(center, dictionary, 10);
          writer.write("\tTop Terms: ");
          writer.write(topTerms);
          writer.write('\n');
        }
        
        List<String> points = clusterIdToPoints.get(String.valueOf(value
            .getId()));
        if (points != null) {
          writer.write("\tPoints: ");
          for (Iterator<String> iterator = points.iterator(); iterator
              .hasNext();) {
            String point = iterator.next();
            writer.append(point);
            if (iterator.hasNext()) {
              writer.append(", ");
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
  
  public Map<String,List<String>> getClusterIdToPoints() {
    return clusterIdToPoints;
  }
  
  public String getTermDictionary() {
    return termDictionary;
  }
  
  public void setTermDictionary(String termDictionary, String dictionaryType) {
    this.termDictionary = termDictionary;
    this.dictionaryFormat = dictionaryType;
  }
  
  public static void main(String[] args) throws IOException,
                                        IllegalAccessException,
                                        InstantiationException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option seqOpt = obuilder.withLongName("seqFileDir").withRequired(false)
        .withArgument(
          abuilder.withName("seqFileDir").withMinimum(1).withMaximum(1)
              .create()).withDescription(
          "The directory containing Sequence Files for the Clusters")
        .withShortName("s").create();
    Option outputOpt = obuilder.withLongName("output").withRequired(false)
        .withArgument(
          abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "The output file.  If not specified, dumps to the console")
        .withShortName("o").create();
    Option substringOpt = obuilder
        .withLongName("substring")
        .withRequired(false)
        .withArgument(
          abuilder.withName("substring").withMinimum(1).withMaximum(1).create())
        .withDescription("The number of chars of the asFormatString() to print")
        .withShortName("b").create();
    Option centroidJSonOpt = obuilder
        .withLongName("json")
        .withRequired(false)
        .withDescription(
          "Output the centroid as JSON.  Otherwise it substitues in the terms for vector cell entries")
        .withShortName("j").create();
    Option pointsOpt = obuilder
        .withLongName("pointsDir")
        .withRequired(false)
        .withArgument(
          abuilder.withName("pointsDir").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "The directory containing points sequence files mapping input vectors to their cluster.  "
              + "If specified, then the program will output the points associated with a cluster")
        .withShortName("p").create();
    Option dictOpt = obuilder.withLongName("dictionary").withRequired(false)
        .withArgument(
          abuilder.withName("dictionary").withMinimum(1).withMaximum(1)
              .create()).withDescription("The dictionary file. ")
        .withShortName("d").create();
    Option dictTypeOpt = obuilder.withLongName("dictionaryType").withRequired(
      false).withArgument(
      abuilder.withName("dictionaryType").withMinimum(1).withMaximum(1)
          .create()).withDescription(
      "The dictionary file type (text|sequencefile)").withShortName("dt")
        .create();
    Option helpOpt = obuilder.withLongName("help").withDescription(
      "Print out help").withShortName("h").create();
    
    Group group = gbuilder.withName("Options").withOption(helpOpt).withOption(
      seqOpt).withOption(outputOpt).withOption(substringOpt).withOption(
      pointsOpt).withOption(centroidJSonOpt).withOption(dictOpt).withOption(
      dictTypeOpt).create();
    
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
      String seqFileDir = cmdLine.getValue(seqOpt).toString();
      String termDictionary = null;
      if (cmdLine.hasOption(dictOpt)) {
        termDictionary = cmdLine.getValue(dictOpt).toString();
      }
      
      String pointsDir = null;
      if (cmdLine.hasOption(pointsOpt)) {
        pointsDir = cmdLine.getValue(pointsOpt).toString();
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
      if (sub > 0) {
        clusterDumper.setSubString(sub);
      }
      clusterDumper.printClusters();
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }
  
  private void setUseJSON(boolean json) {
    this.useJSON = json;
  }
  
  private static Map<String,List<String>> readPoints(String pointsPathDir,
                                                     JobConf conf) throws IOException {
    SortedMap<String,List<String>> result = new TreeMap<String,List<String>>();
    
    File[] children = new File(pointsPathDir).listFiles(new FilenameFilter() {
      @Override
      public boolean accept(File file, String name) {
        return name.endsWith(".crc") == false;
      }
    });
    
    for (File file : children) {
      if (!file.isFile()) {
        continue;
      }
      String pointsPath = file.getAbsolutePath();
      Path path = new Path(pointsPath);
      FileSystem fs = FileSystem.get(path.toUri(), conf);
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
      try {
        Text key = (Text) reader.getKeyClass().newInstance();
        Text value = (Text) reader.getValueClass().newInstance();
        while (reader.next(key, value)) {
          // value is the cluster id as an int, key is the name/id of the
          // vector, but that doesn't matter because we only care about printing
          // it
          String clusterId = value.toString();
          List<String> pointList = result.get(clusterId);
          if (pointList == null) {
            pointList = new ArrayList<String>();
            result.put(clusterId, pointList);
          }
          pointList.add(key.toString());
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
    public int index = -1;
    public double weight = 0;
    
    TermIndexWeight(int index, double weight) {
      this.index = index;
      this.weight = weight;
    }
  }
  
  private static String getTopFeatures(Vector vector,
                                       String[] dictionary,
                                       int numTerms) {
    
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
    
    List<String> topTerms = new LinkedList<String>();
    
    for (int i = 0; i < vectorTerms.size() && i < numTerms; i++) {
      int index = vectorTerms.get(i).index;
      String dictTerm = dictionary[index];
      if (dictTerm == null) {
        log.error("Dictionary entry missing for {}", index);
        continue;
      }
      topTerms.add(dictTerm);
    }
    
    StringBuilder sb = new StringBuilder();
    for (Iterator<String> iterator = topTerms.iterator(); iterator.hasNext();) {
      String term = iterator.next();
      sb.append(term);
      if (iterator.hasNext()) {
        sb.append(", ");
      }
    }
    return sb.toString();
  }
  
}
