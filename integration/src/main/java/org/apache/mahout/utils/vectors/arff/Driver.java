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

package org.apache.mahout.utils.vectors.arff;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.Writer;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.utils.vectors.io.SequenceFileVectorWriter;
import org.apache.mahout.utils.vectors.io.VectorWriter;
import org.codehaus.jackson.map.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Driver {

  private static final Logger log = LoggerFactory.getLogger(Driver.class);

  /** used for JSON serialization/deserialization */
  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

  private Driver() { }
  
  public static void main(String[] args) throws IOException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = obuilder
        .withLongName("input")
        .withRequired(true)
        .withArgument(abuilder.withName("input").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "The file or directory containing the ARFF files.  If it is a directory, all .arff files will be converted")
        .withShortName("d").create();
    
    Option outputOpt = obuilder.withLongName("output").withRequired(true).withArgument(
      abuilder.withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
      "The output directory.  Files will have the same name as the input, but with the extension .mvc")
        .withShortName("o").create();
    
    Option maxOpt = obuilder.withLongName("max").withRequired(false).withArgument(
      abuilder.withName("max").withMinimum(1).withMaximum(1).create()).withDescription(
      "The maximum number of vectors to output.  If not specified, then it will loop over all docs")
        .withShortName("m").create();
    
    Option dictOutOpt = obuilder.withLongName("dictOut").withRequired(true).withArgument(
      abuilder.withName("dictOut").withMinimum(1).withMaximum(1).create()).withDescription(
      "The file to output the label bindings").withShortName("t").create();
    
    Option jsonDictonaryOpt = obuilder.withLongName("json-dictonary").withRequired(false)
            .withDescription("Write dictonary in JSON format").withShortName("j").create();
    
    Option delimiterOpt = obuilder.withLongName("delimiter").withRequired(false).withArgument(
      abuilder.withName("delimiter").withMinimum(1).withMaximum(1).create()).withDescription(
      "The delimiter for outputing the dictionary").withShortName("l").create();
    
    Option helpOpt = obuilder.withLongName("help").withDescription("Print out help").withShortName("h")
        .create();
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(maxOpt)
        .withOption(helpOpt).withOption(dictOutOpt).withOption(jsonDictonaryOpt).withOption(delimiterOpt)
        .create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      
      if (cmdLine.hasOption(helpOpt)) {
        
        CommandLineUtil.printHelp(group);
        return;
      }
      if (cmdLine.hasOption(inputOpt)) { // Lucene case
        File input = new File(cmdLine.getValue(inputOpt).toString());
        long maxDocs = Long.MAX_VALUE;
        if (cmdLine.hasOption(maxOpt)) {
          maxDocs = Long.parseLong(cmdLine.getValue(maxOpt).toString());
        }
        if (maxDocs < 0) {
          throw new IllegalArgumentException("maxDocs must be >= 0");
        }
        String outDir = cmdLine.getValue(outputOpt).toString();
        log.info("Output Dir: {}", outDir);

        String delimiter = cmdLine.hasOption(delimiterOpt) ? cmdLine.getValue(delimiterOpt).toString() : "\t";
        File dictOut = new File(cmdLine.getValue(dictOutOpt).toString());
        boolean jsonDictonary = cmdLine.hasOption(jsonDictonaryOpt);
        ARFFModel model = new MapBackedARFFModel();
        if (input.exists() && input.isDirectory()) {
          File[] files = input.listFiles(new FilenameFilter() {
            @Override
            public boolean accept(File file, String name) {
              return name.endsWith(".arff");
            }
          });
          
          for (File file : files) {
            writeFile(outDir, file, maxDocs, model, dictOut, delimiter, jsonDictonary);
          }
        } else {
          writeFile(outDir, input, maxDocs, model, dictOut, delimiter, jsonDictonary);
        }
      }
      
    } catch (OptionException e) {
      log.error("Exception", e);
      CommandLineUtil.printHelp(group);
    }
  }
  
  protected static void writeLabelBindings(File dictOut, ARFFModel arffModel, String delimiter, boolean jsonDictonary)
    throws IOException {
    Writer writer = Files.newWriterSupplier(dictOut, Charsets.UTF_8, true).getOutput();
    try {
      if (jsonDictonary) {
        writeLabelBindingsJSON(writer, arffModel);
      } else {
        writeLabelBindings(writer, arffModel, delimiter);
      }
    } finally {
      Closeables.close(writer, false);
    }
  }

  protected static void writeLabelBindingsJSON(Writer writer, ARFFModel arffModel) throws IOException  {

    // Turn the map of labels into a list order by order of appearance
    List<Entry<String, Integer>> attributes = Lists.newArrayList();
    attributes.addAll(arffModel.getLabelBindings().entrySet());
    Collections.sort(attributes, new Comparator<Map.Entry<String, Integer>>() {
      @Override
      public int compare(Entry<String, Integer> t, Entry<String, Integer> t1) {
        return t.getValue().compareTo(t1.getValue());
      }
    });
    
    // write a map for each object
    List<Map<String, Object>> jsonObjects = Lists.newLinkedList();
    for (int i = 0; i < attributes.size(); i++) { 
      
      Entry<String, Integer> modelRepresentation = attributes.get(i);
      Map<String, Object> jsonRepresentation = Maps.newHashMap();
      jsonObjects.add(jsonRepresentation);
      // the last one is the class label
      jsonRepresentation.put("label", i < (attributes.size() - 1) ? String.valueOf(false) : String.valueOf(true));
      String attribute = modelRepresentation.getKey();
      jsonRepresentation.put("attribute", attribute);
      Map<String, Integer> nominalValues = arffModel.getNominalMap().get(attribute);

      if (nominalValues != null) {
        String[] values = nominalValues.keySet().toArray(new String[1]);

        jsonRepresentation.put("values", values);
        jsonRepresentation.put("type", "categorical");
      } else {
        jsonRepresentation.put("type", "numerical");
      }
    }
    writer.write(OBJECT_MAPPER.writeValueAsString(jsonObjects));
  }

  protected static void writeLabelBindings(Writer writer, ARFFModel arffModel, String delimiter) throws IOException {

    Map<String, Integer> labels = arffModel.getLabelBindings();
    writer.write("Label bindings for Relation " + arffModel.getRelation() + '\n');
    for (Map.Entry<String, Integer> entry : labels.entrySet()) {
      writer.write(entry.getKey());
      writer.write(delimiter);
      writer.write(String.valueOf(entry.getValue()));
      writer.write('\n');
    }
    writer.write('\n');
    writer.write("Values for nominal attributes\n");
    // emit allowed values for NOMINAL/categorical/enumerated attributes
    Map<String, Map<String, Integer>> nominalMap = arffModel.getNominalMap();
    // how many nominal attributes
    writer.write(String.valueOf(nominalMap.size()) + "\n");

    for (Entry<String, Map<String, Integer>> entry : nominalMap.entrySet()) {
      // the label of this attribute
      writer.write(entry.getKey() + "\n");
      Set<Entry<String, Integer>> attributeValues = entry.getValue().entrySet();
      // how many values does this attribute have
      writer.write(attributeValues.size() + "\n");
      for (Map.Entry<String, Integer> value : attributeValues) {
        // the value and the value index
        writer.write(String.format("%s%s%s\n", value.getKey(), delimiter, value.getValue().toString()));
      }
    }
  }
  
  protected static void writeFile(String outDir,
                                File file,
                                long maxDocs,
                                ARFFModel arffModel,
                                File dictOut,
                                String delimiter,
                                boolean jsonDictonary) throws IOException {
    log.info("Converting File: {}", file);
    ARFFModel model = new MapBackedARFFModel(arffModel.getWords(), arffModel.getWordCount() + 1, arffModel
        .getNominalMap());
    Iterable<Vector> iteratable = new ARFFVectorIterable(file, model);
    String outFile = outDir + '/' + file.getName() + ".mvc";
    
    VectorWriter vectorWriter = getSeqFileWriter(outFile);
    try {
      long numDocs = vectorWriter.write(iteratable, maxDocs);
      writeLabelBindings(dictOut, model, delimiter, jsonDictonary);
      log.info("Wrote: {} vectors", numDocs);
    } finally {
      Closeables.close(vectorWriter, false);
    }
  }
  
  private static VectorWriter getSeqFileWriter(String outFile) throws IOException {
    Path path = new Path(outFile);
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer seqWriter = SequenceFile.createWriter(fs, conf, path, LongWritable.class,
      VectorWritable.class);
    return new SequenceFileVectorWriter(seqWriter);
  }
  
}
