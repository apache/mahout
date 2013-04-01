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

package org.apache.mahout.classifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.google.common.base.Charsets;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;

import com.google.common.collect.Lists;

/**
 * Export a ConfusionMatrix in various text formats: ToString version Grayscale HTML table Summary HTML table
 * Table of counts all with optional HTML wrappers
 * 
 * Input format: Hadoop SequenceFile with Text key and MatrixWritable value, 1 pair
 * 
 * Intended to consume ConfusionMatrix SequenceFile output by Bayes TestClassifier class
 */
public final class ConfusionMatrixDumper extends AbstractJob {

  private static final String TAB_SEPARATOR = "|";

  // HTML wrapper - default CSS
  private static final String HEADER = "<html>"
                                       + "<head>\n"
                                       + "<title>TITLE</title>\n"
                                       + "</head>"
                                       + "<body>\n"
                                       + "<style type='text/css'> \n"
                                       + "table\n"
                                       + "{\n"
                                       + "border:3px solid black; text-align:left;\n"
                                       + "}\n"
                                       + "th.normalHeader\n"
                                       + "{\n"
                                       + "border:1px solid black;border-collapse:collapse;text-align:center;"
                                       + "background-color:white\n"
                                       + "}\n"
                                       + "th.tallHeader\n"
                                       + "{\n"
                                       + "border:1px solid black;border-collapse:collapse;text-align:center;"
                                       + "background-color:white; height:6em\n"
                                       + "}\n"
                                       + "tr.label\n"
                                       + "{\n"
                                       + "border:1px solid black;border-collapse:collapse;text-align:center;"
                                       + "background-color:white\n"
                                       + "}\n"
                                       + "tr.row\n"
                                       + "{\n"
                                       + "border:1px solid gray;text-align:center;background-color:snow\n"
                                       + "}\n"
                                       + "td\n"
                                       + "{\n"
                                       + "min-width:2em\n"
                                       + "}\n"
                                       + "td.cell\n"
                                       + "{\n"
                                       + "border:1px solid black;text-align:right;background-color:snow\n"
                                       + "}\n"
                                       + "td.empty\n"
                                       + "{\n"
                                       + "border:0px;text-align:right;background-color:snow\n"
                                       + "}\n"
                                       + "td.white\n"
                                       + "{\n"
                                       + "border:0px solid black;text-align:right;background-color:white\n"
                                       + "}\n"
                                       + "td.black\n"
                                       + "{\n"
                                       + "border:0px solid red;text-align:right;background-color:black\n"
                                       + "}\n"
                                       + "td.gray1\n"
                                       + "{\n"
                                       + "border:0px solid green;text-align:right; background-color:LightGray\n"
                                       + "}\n" + "td.gray2\n" + "{\n"
                                       + "border:0px solid blue;text-align:right;background-color:gray\n"
                                       + "}\n" + "td.gray3\n" + "{\n"
                                       + "border:0px solid red;text-align:right;background-color:DarkGray\n"
                                       + "}\n" + "th" + "{\n" + "        text-align: center;\n"
                                       + "        vertical-align: bottom;\n"
                                       + "        padding-bottom: 3px;\n" + "        padding-left: 5px;\n"
                                       + "        padding-right: 5px;\n" + "}\n" + "     .verticalText\n"
                                       + "      {\n" + "        text-align: center;\n"
                                       + "        vertical-align: middle;\n" + "        width: 20px;\n"
                                       + "        margin: 0px;\n" + "        padding: 0px;\n"
                                       + "        padding-left: 3px;\n" + "        padding-right: 3px;\n"
                                       + "        padding-top: 10px;\n" + "        white-space: nowrap;\n"
                                       + "        -webkit-transform: rotate(-90deg); \n"
                                       + "        -moz-transform: rotate(-90deg);         \n" + "      };\n"
                                       + "</style>\n";
  private static final String FOOTER = "</html></body>";
  
  // CSS style names.
  private static final String CSS_TABLE = "table";
  private static final String CSS_LABEL = "label";
  private static final String CSS_TALL_HEADER = "tall";
  private static final String CSS_VERTICAL = "verticalText";
  private static final String CSS_CELL = "cell";
  private static final String CSS_EMPTY = "empty";
  private static final String[] CSS_GRAY_CELLS = {"white", "gray1", "gray2", "gray3", "black"};
  
  private ConfusionMatrixDumper() {}
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ConfusionMatrixDumper(), args);
  }
  
  @Override
  public int run(String[] args) throws IOException {
    addInputOption();
    addOption("output", "o", "Output path", null); // AbstractJob output feature requires param
    addOption(DefaultOptionCreator.overwriteOption().create());
    addFlag("html", null, "Create complete HTML page");
    addFlag("text", null, "Dump simple text");
    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }
    
    Path inputPath = getInputPath();
    String outputFile = hasOption("output") ? getOption("output") : null;
    boolean text = parsedArgs.containsKey("--text");
    boolean wrapHtml = parsedArgs.containsKey("--html");
    PrintStream out = getPrintStream(outputFile);
    if (text) {
      exportText(inputPath, out);
    } else {
      exportTable(inputPath, out, wrapHtml);
    }
    out.flush();
    if (out != System.out) {
      out.close();
    }
    return 0;
  }
  
  private static void exportText(Path inputPath, PrintStream out) throws IOException {
    MatrixWritable mw = new MatrixWritable();
    Text key = new Text();
    readSeqFile(inputPath, key, mw);
    Matrix m = mw.get();
    ConfusionMatrix cm = new ConfusionMatrix(m);
    out.println(String.format("%-40s", "Label") + TAB_SEPARATOR + String.format("%-10s", "Total")
                + TAB_SEPARATOR + String.format("%-10s", "Correct") + TAB_SEPARATOR
                + String.format("%-6s", "%") + TAB_SEPARATOR);
    out.println(String.format("%-70s", "-").replace(' ', '-'));
    List<String> labels = stripDefault(cm);
    for (String label : labels) {
      int correct = cm.getCorrect(label);
      double accuracy = cm.getAccuracy(label);
      int count = getCount(cm, label);
      out.println(String.format("%-40s", label) + TAB_SEPARATOR + String.format("%-10s", count)
                  + TAB_SEPARATOR + String.format("%-10s", correct) + TAB_SEPARATOR
                  + String.format("%-6s", (int) Math.round(accuracy)) + TAB_SEPARATOR);
    }
    out.println(String.format("%-70s", "-").replace(' ', '-'));
    out.println(cm.toString());
  }
  
  private static void exportTable(Path inputPath, PrintStream out, boolean wrapHtml) throws IOException {
    MatrixWritable mw = new MatrixWritable();
    Text key = new Text();
    readSeqFile(inputPath, key, mw);
    String fileName = inputPath.getName();
    fileName = fileName.substring(fileName.lastIndexOf('/') + 1, fileName.length());
    Matrix m = mw.get();
    ConfusionMatrix cm = new ConfusionMatrix(m);
    if (wrapHtml) {
      printHeader(out, fileName);
    }
    out.println("<p/>");
    printSummaryTable(cm, out);
    out.println("<p/>");
    printGrayTable(cm, out);
    out.println("<p/>");
    printCountsTable(cm, out);
    out.println("<p/>");
    printTextInBox(cm, out);
    out.println("<p/>");
    if (wrapHtml) {
      printFooter(out);
    }
  }
  
  private static List<String> stripDefault(ConfusionMatrix cm) {
    List<String> stripped = Lists.newArrayList(cm.getLabels().iterator());
    String defaultLabel = cm.getDefaultLabel();
    int unclassified = cm.getTotal(defaultLabel);
    if (unclassified > 0) {
      return stripped;
    }
    stripped.remove(defaultLabel);
    return stripped;
  }
  
  // TODO: test - this should work with HDFS files
  private static void readSeqFile(Path path, Text key, MatrixWritable m) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);
    reader.next(key, m);
  }
  
  // TODO: test - this might not work with HDFS files?
  // after all, it does no seeks
  private static PrintStream getPrintStream(String outputFilename) throws IOException {
    if (outputFilename != null) {
      File outputFile = new File(outputFilename);
      if (outputFile.exists()) {
        outputFile.delete();
      }
      outputFile.createNewFile();
      OutputStream os = new FileOutputStream(outputFile);
      return new PrintStream(os, false, Charsets.UTF_8.displayName());
    } else {
      return System.out;
    }
  }
  
  private static int getLabelTotal(ConfusionMatrix cm, String rowLabel) {
    Iterator<String> iter = cm.getLabels().iterator();
    int count = 0;
    while (iter.hasNext()) {
      count += cm.getCount(rowLabel, iter.next());
    }
    return count;
  }
  
  // HTML generator code
  
  private static void printTextInBox(ConfusionMatrix cm, PrintStream out) {
    out.println("<div style='width:90%;overflow:scroll;'>");
    out.println("<pre>");
    out.println(cm.toString());
    out.println("</pre>");
    out.println("</div>");
  }
  
  public static void printSummaryTable(ConfusionMatrix cm, PrintStream out) {
    format("<table class='%s'>\n", out, CSS_TABLE);
    format("<tr class='%s'>", out, CSS_LABEL);
    out.println("<td>Label</td><td>Total</td><td>Correct</td><td>%</td>");
    out.println("</tr>");
    List<String> labels = stripDefault(cm);
    for (String label : labels) {
      printSummaryRow(cm, out, label);
    }
    out.println("</table>");
  }
  
  private static void printSummaryRow(ConfusionMatrix cm, PrintStream out, String label) {
    format("<tr class='%s'>", out, CSS_CELL);
    int correct = cm.getCorrect(label);
    double accuracy = cm.getAccuracy(label);
    int count = getCount(cm, label);
    format("<td class='%s'>%s</td><td>%d</td><td>%d</td><td>%d</td>", out, CSS_CELL, label, count, correct,
      (int) Math.round(accuracy));
    out.println("</tr>");
  }
  
  private static int getCount(ConfusionMatrix cm, String label) {
    int count = 0;
    for (String s : cm.getLabels()) {
      count += cm.getCount(label, s);
    }
    return count;
  }
  
  public static void printGrayTable(ConfusionMatrix cm, PrintStream out) {
    format("<table class='%s'>\n", out, CSS_TABLE);
    printCountsHeader(cm, out, true);
    printGrayRows(cm, out);
    out.println("</table>");
  }
  
  /**
   * Print each value in a four-value grayscale based on count/max. Gives a mostly white matrix with grays in
   * misclassified, and black in diagonal. TODO: Using the sqrt(count/max) as the rating is more stringent
   */
  private static void printGrayRows(ConfusionMatrix cm, PrintStream out) {
    List<String> labels = stripDefault(cm);
    for (String label : labels) {
      printGrayRow(cm, out, labels, label);
    }
  }
  
  private static void printGrayRow(ConfusionMatrix cm,
                                   PrintStream out,
                                   Iterable<String> labels,
                                   String rowLabel) {
    format("<tr class='%s'>", out, CSS_LABEL);
    format("<td>%s</td>", out, rowLabel);
    int total = getLabelTotal(cm, rowLabel);
    for (String columnLabel : labels) {
      printGrayCell(cm, out, total, rowLabel, columnLabel);
    }
    out.println("</tr>");
  }
  
  // assign white/light/medium/dark to 0,1/4,1/2,3/4 of total number of inputs
  // assign black to count = total, meaning complete success
  // alternative rating is to use sqrt(total) instead of total - this is more drastic
  private static void printGrayCell(ConfusionMatrix cm,
                                    PrintStream out,
                                    int total,
                                    String rowLabel,
                                    String columnLabel) {
    
    int count = cm.getCount(rowLabel, columnLabel);
    if (count == 0) {
      out.format("<td class='%s'/>", CSS_EMPTY);
    } else {
      // 0 is white, full is black, everything else gray
      int rating = (int) ((count / (double) total) * 4);
      String css = CSS_GRAY_CELLS[rating];
      format("<td class='%s' title='%s'>%s</td>", out, css, columnLabel, count);
    }
  }
  
  public static void printCountsTable(ConfusionMatrix cm, PrintStream out) {
    format("<table class='%s'>\n", out, CSS_TABLE);
    printCountsHeader(cm, out, false);
    printCountsRows(cm, out);
    out.println("</table>");
  }
  
  private static void printCountsRows(ConfusionMatrix cm, PrintStream out) {
    List<String> labels = stripDefault(cm);
    for (String label : labels) {
      printCountsRow(cm, out, labels, label);
    }
  }
  
  private static void printCountsRow(ConfusionMatrix cm,
                                     PrintStream out,
                                     Iterable<String> labels,
                                     String rowLabel) {
    out.println("<tr>");
    format("<td class='%s'>%s</td>", out, CSS_LABEL, rowLabel);
    for (String columnLabel : labels) {
      printCountsCell(cm, out, rowLabel, columnLabel);
    }
    out.println("</tr>");
  }
  
  private static void printCountsCell(ConfusionMatrix cm, PrintStream out, String rowLabel, String columnLabel) {
    int count = cm.getCount(rowLabel, columnLabel);
    String s = count == 0 ? "" : Integer.toString(count);
    format("<td class='%s' title='%s'>%s</td>", out, CSS_CELL, columnLabel, s);
  }
  
  private static void printCountsHeader(ConfusionMatrix cm, PrintStream out, boolean vertical) {
    List<String> labels = stripDefault(cm);
    int longest = getLongestHeader(labels);
    if (vertical) {
      // do vertical - rotation is a bitch
      out.format("<tr class='%s' style='height:%dem'><th>&nbsp;</th>%n", CSS_TALL_HEADER, longest / 2);
      for (String label : labels) {
        out.format("<th><div class='%s'>%s</div></th>", CSS_VERTICAL, label);
      }
      out.println("</tr>");
    } else {
      // header - empty cell in upper left
      out.format("<tr class='%s'><td class='%s'></td>%n", CSS_TABLE, CSS_LABEL);
      for (String label : labels) {
        out.format("<td>%s</td>", label);
      }
      out.format("</tr>");
    }
  }
  
  private static int getLongestHeader(Iterable<String> labels) {
    int max = 0;
    for (String label : labels) {
      max = Math.max(label.length(), max);
    }
    return max;
  }
  
  private static void format(String format, PrintStream out, Object... args) {
    String format2 = String.format(format, args);
    out.println(format2);
  }
  
  public static void printHeader(PrintStream out, CharSequence title) {
    out.println(HEADER.replace("TITLE", title));
  }
  
  public static void printFooter(PrintStream out) {
    out.println(FOOTER);
  }
  
}
