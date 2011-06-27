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

package org.apache.mahout.clustering.minhash;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public final class LastfmDataConverter {

  private static final Pattern TAB_PATTERN = Pattern.compile("\t");

  // we are clustering similar featureIdxs on the following dataset
  // http://www.iua.upf.es/~ocelma/MusicRecommendationDataset/index.html
  //
  // Preparation of the data set means gettting the dataset to a format which
  // can
  // be read by the min hash algorithm;
  //
  enum Lastfm {
    USERS_360K(17559530),
    USERS_1K(19150868);
    private final int totalRecords;
    Lastfm(int totalRecords) {
      this.totalRecords = totalRecords;
    }
    int getTotalRecords() {
      return totalRecords;
    }
  }

  private LastfmDataConverter() {
  }

  private static String usedMemory() {
    Runtime runtime = Runtime.getRuntime();
    return "Used Memory: [" + (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024) + " MB] ";
  }

  /* Get the feature from the parsed record */
  private static String getFeature(String[] fields, Lastfm dataSet) {
    if (dataSet == Lastfm.USERS_360K) {
      return fields[0];
    } else {
      return fields[2];
    }
  }

  /* Get the item from the parsed record */
  private static String getItem(String[] fields, Lastfm dataSet) {
    if (dataSet == Lastfm.USERS_360K) {
      return fields[2];
    } else {
      return fields[0];
    }
  }

  /**
   * Reads the LastFm dataset and constructs a Map of (item, features). For 360K
   * Users dataset - (Item=Artist, Feature=User) For 1K Users dataset -
   * (Item=User, Feature=Artist)
   * 
   * @param inputFile
   *          Lastfm dataset file on the local file system.
   * @param dataSet
   *          Type of dataset - 360K Users or 1K Users
   */
  public static Map<String, List<Integer>> convertToItemFeatures(String inputFile, Lastfm dataSet) throws IOException {
    long totalRecords = dataSet.getTotalRecords();
    Map<String, Integer> featureIdxMap = Maps.newHashMap();
    Map<String, List<Integer>> itemFeaturesMap = Maps.newHashMap();
    String msg = usedMemory() + "Converting data to internal vector format: ";
    BufferedReader br = Files.newReader(new File(inputFile), Charsets.UTF_8);
    try {
      System.out.print(msg);
      int prevPercentDone = 1;
      double percentDone = 0.0;
      long parsedRecords = 0;
      String line;
      while ((line = br.readLine()) != null) {
        String[] fields = TAB_PATTERN.split(line);
        String feature = getFeature(fields, dataSet);
        String item = getItem(fields, dataSet);
        // get the featureIdx
        Integer featureIdx = featureIdxMap.get(feature);
        if (featureIdx == null) {
          featureIdx = featureIdxMap.size() + 1;
          featureIdxMap.put(feature, featureIdx);
        }
        // add it to the corresponding feature idx map
        List<Integer> features = itemFeaturesMap.get(item);
        if (features == null) {
          features = Lists.newArrayList();
          itemFeaturesMap.put(item, features);
        }
        features.add(featureIdx);
        parsedRecords++;
        // Update the progress
        percentDone = parsedRecords * 100.0 / totalRecords;
        msg = usedMemory() + "Converting data to internal vector format: ";
        if (percentDone > prevPercentDone) {
          System.out.print('\r' + msg + percentDone + '%');
          prevPercentDone++;
        }
        parsedRecords++;
      }
      msg = usedMemory() + "Converting data to internal vector format: ";
      System.out.print('\r' + msg + percentDone + "% Completed\n");
    } finally {
      Closeables.closeQuietly(br);
    }
    return itemFeaturesMap;
  }

  /**
   * Converts each record in (item,features) map into Mahout vector format and
   * writes it into sequencefile for minhash clustering
   */
  public static boolean writeToSequenceFile(Map<String, List<Integer>> itemFeaturesMap, Path outputPath)
    throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    fs.mkdirs(outputPath.getParent());
    long totalRecords = itemFeaturesMap.size();
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, outputPath, Text.class, VectorWritable.class);
    try {
      String msg = "Now writing vectorized data in sequence file format: ";
      System.out.print(msg);

      Text itemWritable = new Text();
      VectorWritable featuresWritable = new VectorWritable();

      int doneRecords = 0;
      int prevPercentDone = 1;

      for (Map.Entry<String, List<Integer>> itemFeature : itemFeaturesMap.entrySet()) {
        int numfeatures = itemFeature.getValue().size();
        itemWritable.set(itemFeature.getKey());
        Vector featureVector = new SequentialAccessSparseVector(numfeatures);
        int i = 0;
        for (Integer feature : itemFeature.getValue()) {
          featureVector.setQuick(i++, feature);
        }
        featuresWritable.set(featureVector);
        writer.append(itemWritable, featuresWritable);
        // Update the progress
        double percentDone = ++doneRecords * 100.0 / totalRecords;
        if (percentDone > prevPercentDone) {
          System.out.print('\r' + msg + percentDone + "% " + (percentDone >= 100 ? "Completed\n" : ""));
          prevPercentDone++;
        }
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
    return true;
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 3) {
      System.out.println("[Usage]: LastfmDataConverter <input> <output> <dataset>");
      System.out.println("   <input>: Absolute path to the local file [usersha1-artmbid-artname-plays.tsv] ");
      System.out.println("  <output>: Absolute path to the HDFS output file");
      System.out.println(" <dataset>: Either of the two Lastfm public datasets. "
          + "Must be either 'Users360K' or 'Users1K'");
      System.out.println("Note:- Hadoop configuration pointing to HDFS namenode should be in classpath");
      return;
    }
    Lastfm dataSet = Lastfm.valueOf(args[2]);
    Map<String, List<Integer>> itemFeatures = convertToItemFeatures(args[0], dataSet);
    if (itemFeatures.isEmpty()) {
      throw new IllegalStateException("Error converting the data file: [" + args[0] + ']');
    }
    Path output = new Path(args[1]);
    boolean status = writeToSequenceFile(itemFeatures, output);
    if (status) {
      System.out.println("Data converted and written successfully to HDFS location: [" + output + ']');
    } else {
      System.err.println("Error writing the converted data to HDFS location: [" + output + ']');
    }
  }
}
