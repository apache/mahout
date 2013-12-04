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

package org.apache.mahout.utils.vectors;

import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Iterator;
import java.util.Set;

import com.google.common.base.Charsets;
import com.google.common.collect.Sets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.Utils.OutputFileUtils.OutputFilesFilter;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Can read in a {@link org.apache.hadoop.io.SequenceFile} of {@link Vector}s and dump
 * out the results using {@link Vector#asFormatString()} to either the console or to a
 * file.
 */
public final class VectorDumper extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(VectorDumper.class);

  private VectorDumper() {}

  @Override
  public int run(String[] args) throws Exception {
    /**
     Option seqOpt = obuilder.withLongName("seqFile").withRequired(false).withArgument(
     abuilder.withName("seqFile").withMinimum(1).withMaximum(1).create()).withDescription(
     "The Sequence File containing the Vectors").withShortName("s").create();
     Option dirOpt = obuilder.withLongName("seqDirectory").withRequired(false).withArgument(
     abuilder.withName("seqDirectory").withMinimum(1).withMaximum(1).create())
     .withDescription("The directory containing Sequence File of Vectors")
     .withShortName("d").create();
     */
    addInputOption();
    addOutputOption();
    addOption("useKey", "u", "If the Key is a vector than dump that instead");
    addOption("printKey", "p", "Print out the key as well, delimited by tab (or the value if useKey is true");
    addOption("dictionary", "d", "The dictionary file.", false);
    addOption("dictionaryType", "dt", "The dictionary file type (text|seqfile)", false);
    addOption("csv", "c", "Output the Vector as CSV.  Otherwise it substitutes in the terms for vector cell entries");
    addOption("namesAsComments", "n", "If using CSV output, optionally add a comment line for each NamedVector "
        + "(if the vector is one) printing out the name");
    addOption("nameOnly", "N", "Use the name as the value for each NamedVector (skip other vectors)");
    addOption("sortVectors", "sort", "Sort output key/value pairs of the vector entries in abs magnitude "
        + "descending order");
    addOption("quiet", "q", "Print only file contents");
    addOption("sizeOnly", "sz", "Dump only the size of the vector");
    addOption("numItems", "ni", "Output at most <n> vecors", false);
    addOption("vectorSize", "vs", "Truncate vectors to <vs> length when dumping (most useful when in"
            + " conjunction with -sort", false);
    addOption(buildOption("filter", "fi", "Only dump out those vectors whose name matches the filter." 
            + "  Multiple items may be specified by repeating the argument.", true, 1, Integer.MAX_VALUE, false, null));

    if (parseArguments(args, false, true) == null) {
      return -1;
    }

    Path[] pathArr;
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    Path input = getInputPath();
    FileStatus fileStatus = fs.getFileStatus(input);
    if (fileStatus.isDir()) {
      pathArr = FileUtil.stat2Paths(fs.listStatus(input, new OutputFilesFilter()));
    } else {
      FileStatus[] inputPaths = fs.globStatus(input);
      pathArr = new Path[inputPaths.length];
      int i = 0;
      for (FileStatus fstatus : inputPaths) {
        pathArr[i++] = fstatus.getPath();
      }
    }


    String dictionaryType = getOption("dictionaryType", "text");

    boolean sortVectors = hasOption("sortVectors");
    boolean quiet = hasOption("quiet");
    if (!quiet) {
      log.info("Sort? {}", sortVectors);
    }

    String[] dictionary = null;
    if (hasOption("dictionary")) {
      String dictFile = getOption("dictionary");
      if ("text".equals(dictionaryType)) {
        dictionary = VectorHelper.loadTermDictionary(new File(dictFile));
      } else if ("sequencefile".equals(dictionaryType)) {
        dictionary = VectorHelper.loadTermDictionary(conf, dictFile);
      } else {
        //TODO: support Lucene's FST as a dictionary type
        throw new IOException("Invalid dictionary type: " + dictionaryType);
      }
    }

    Set<String> filters;
    if (hasOption("filter")) {
      filters = Sets.newHashSet(getOptions("filter"));
    } else {
      filters = null;
    }

    boolean useCSV = hasOption("csv");

    boolean sizeOnly = hasOption("sizeOnly");
    boolean nameOnly = hasOption("nameOnly");
    boolean namesAsComments = hasOption("namesAsComments");
    boolean transposeKeyValue = hasOption("vectorAsKey");
    Writer writer;
    boolean shouldClose;
    File output = getOutputFile();
    if (output != null) {
      shouldClose = true;
      log.info("Output file: {}", output);
      Files.createParentDirs(output);
      writer = Files.newWriter(output, Charsets.UTF_8);
    } else {
      shouldClose = false;
      writer = new OutputStreamWriter(System.out, Charsets.UTF_8);
    }
    try {
      boolean printKey = hasOption("printKey");
      if (useCSV && dictionary != null) {
        writer.write("#");
        for (int j = 0; j < dictionary.length; j++) {
          writer.write(dictionary[j]);
          if (j < dictionary.length - 1) {
            writer.write(',');
          }
        }
        writer.write('\n');
      }
      Long numItems = null;
      if (hasOption("numItems")) {
        numItems = Long.parseLong(getOption("numItems"));
        if (quiet) {
          writer.append("#Max Items to dump: ").append(String.valueOf(numItems)).append('\n');
        }
      }
      int maxIndexesPerVector = hasOption("vectorSize")
              ? Integer.parseInt(getOption("vectorSize"))
              : Integer.MAX_VALUE;
      long itemCount = 0;
      int fileCount = 0;
      for (Path path : pathArr) {
        if (numItems != null && numItems <= itemCount) {
          break;
        }
        if (quiet) {
          log.info("Processing file '{}' ({}/{})", path, ++fileCount, pathArr.length);
        }
        SequenceFileIterable<Writable, Writable> iterable =
                new SequenceFileIterable<Writable, Writable>(path, true, conf);
        Iterator<Pair<Writable, Writable>> iterator = iterable.iterator();
        long i = 0;
        while (iterator.hasNext() && (numItems == null || itemCount < numItems)) {
          Pair<Writable, Writable> record = iterator.next();
          Writable keyWritable = record.getFirst();
          Writable valueWritable = record.getSecond();
          if (printKey) {
            Writable notTheVectorWritable = transposeKeyValue ? valueWritable : keyWritable;
            writer.write(notTheVectorWritable.toString());
            writer.write('\t');
          }
          Vector vector;
          try {
            vector = ((VectorWritable)
                    (transposeKeyValue ? keyWritable : valueWritable)).get();
          } catch (ClassCastException e) {
            if ((transposeKeyValue ? keyWritable : valueWritable)
                    instanceof WeightedPropertyVectorWritable) {
              vector =
                  ((WeightedPropertyVectorWritable)
                      (transposeKeyValue ? keyWritable : valueWritable)).getVector();
            } else {
              throw e;
            }
          }
          if (filters != null
                  && vector instanceof NamedVector
                  && !filters.contains(((NamedVector) vector).getName())) {
            //we are filtering out this item, skip
            continue;
          }
          if (sizeOnly) {
            if (vector instanceof NamedVector) {
              writer.write(((NamedVector) vector).getName());
              writer.write(":");
            } else {
              writer.write(String.valueOf(i++));
              writer.write(":");
            }
            writer.write(String.valueOf(vector.size()));
            writer.write('\n');
          } else if (nameOnly) {
            if (vector instanceof NamedVector) {
              writer.write(((NamedVector) vector).getName());
              writer.write('\n');
            }
          } else {
            String fmtStr;
            if (useCSV) {
              fmtStr = VectorHelper.vectorToCSVString(vector, namesAsComments);
            } else {
              fmtStr = VectorHelper.vectorToJson(vector, dictionary, maxIndexesPerVector,
                      sortVectors);
            }
            writer.write(fmtStr);
            writer.write('\n');
          }
          itemCount++;
        }
      }
      writer.flush();
    } finally {
      if (shouldClose) {
        Closeables.close(writer, false);
      }
    }

    return 0;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new VectorDumper(), args);
  }

}
