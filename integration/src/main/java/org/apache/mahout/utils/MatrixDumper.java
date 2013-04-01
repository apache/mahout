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

package org.apache.mahout.utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.List;
import java.util.Map;

import com.google.common.base.Charsets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;

/**
 * Export a Matrix in various text formats:
 *    * CSV file
 * 
 * Input format: Hadoop SequenceFile with Text key and MatrixWritable value, 1 pair
 * TODO:
 *     Needs class for key value- should not hard-code to Text.
 *     Options for row and column headers- stats software can be picky.
 * Assumes only one matrix in a file.
 */
public final class MatrixDumper extends AbstractJob {
  
  private MatrixDumper() { }
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixDumper(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {
    
    addInputOption();
    addOption("output", "o", "Output path", null); // AbstractJob output feature requires param
    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    String outputFile = hasOption("output") ? getOption("output") : null;
    exportCSV(getInputPath(), outputFile, false);
    return 0;
  }
  
  private static void exportCSV(Path inputPath, String outputFile, boolean doLabels) throws IOException {
    SequenceFileValueIterator<MatrixWritable> it =
        new SequenceFileValueIterator<MatrixWritable>(inputPath, true, new Configuration());
    Matrix m = it.next().get();
    it.close();
    PrintStream ps = getPrintStream(outputFile);
    String[] columnLabels = getLabels(m.numCols(), m.getColumnLabelBindings(), "col");
    String[] rowLabels = getLabels(m.numRows(), m.getRowLabelBindings(), "row");
    if (doLabels) {
      ps.print("rowid,");
      ps.print(columnLabels[0]);
      for (int c = 1; c < m.numCols(); c++) {
        ps.print(',' + columnLabels[c]);
      }
      ps.println();
    }
    for (int r = 0; r < m.numRows(); r++) {
      if (doLabels) {
        ps.print(rowLabels[0] + ',');
      }
      ps.print(Double.toString(m.getQuick(r,0)));
      for (int c = 1; c < m.numCols(); c++) {
        ps.print(",");
        ps.print(Double.toString(m.getQuick(r,c)));
      }
      ps.println();
    }
    if (ps != System.out) {
      ps.close();
    }
  }
  
  private static PrintStream getPrintStream(String outputPath) throws IOException {
    if (outputPath == null) {
      return System.out;
    }
    File outputFile = new File(outputPath);
    if (outputFile.exists()) {
      outputFile.delete();
    }
    outputFile.createNewFile();
    OutputStream os = new FileOutputStream(outputFile);
    return new PrintStream(os, false, Charsets.UTF_8.displayName());
  }
  
  /**
   * return the label set, sorted by matrix order
   * if there are no labels, fabricate them using the starter string
   * @param length 
   */
  private static String[] getLabels(int length, Map<String,Integer> labels, String start) {
    if (labels != null) {
      return sortLabels(labels);
    }
    String[] sorted = new String[length];
    for (int i = 1; i <= length; i++) {
      sorted[i] = start + i;
    }
    return sorted;
  }
  
  private static String[] sortLabels(Map<String,Integer> labels) {
    String[] sorted = new String[labels.size()];
    for (Map.Entry<String,Integer> entry : labels.entrySet()) {
      sorted[entry.getValue()] = entry.getKey();
    }
    return sorted;
  }
  
}
