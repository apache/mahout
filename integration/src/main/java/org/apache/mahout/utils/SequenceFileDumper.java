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

package org.apache.mahout.utils;

import java.io.File;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.List;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.Utils.OutputFileUtils.OutputFilesFilter;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterator;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

public final class SequenceFileDumper extends AbstractJob {

  public SequenceFileDumper() {
    setConf(new Configuration());
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("substring", "b", "The number of chars to print out per value", false);
    addOption(buildOption("count", "c", "Report the count only", false, false, null));
    addOption("numItems", "n", "Output at most <n> key value pairs", false);
    addOption(buildOption("facets", "fa", "Output the counts per key.  Note, if there are a lot of unique keys, "
        + "this can take up a fair amount of memory", false, false, null));
    addOption(buildOption("quiet", "q", "Print only file contents.", false, false, null));

    if (parseArguments(args, false, true) == null) {
      return -1;
    }

    Path[] pathArr;
    Configuration conf = new Configuration();
    Path input = getInputPath();
    FileSystem fs = input.getFileSystem(conf);
    if (fs.getFileStatus(input).isDir()) {
      pathArr = FileUtil.stat2Paths(fs.listStatus(input, new OutputFilesFilter()));
    } else {
      pathArr = new Path[1];
      pathArr[0] = input;
    }


    Writer writer;
    boolean shouldClose;
    if (hasOption("output")) {
      shouldClose = true;
      writer = Files.newWriter(new File(getOption("output")), Charsets.UTF_8);
    } else {
      shouldClose = false;
      writer = new OutputStreamWriter(System.out, Charsets.UTF_8);
    }
    try {
      for (Path path : pathArr) {
        if (!hasOption("quiet")) {
          writer.append("Input Path: ").append(String.valueOf(path)).append('\n');
        }

        int sub = Integer.MAX_VALUE;
        if (hasOption("substring")) {
          sub = Integer.parseInt(getOption("substring"));
        }
        boolean countOnly = hasOption("count");
        SequenceFileIterator<?, ?> iterator = new SequenceFileIterator<Writable, Writable>(path, true, conf);
        if (!hasOption("quiet")) {
          writer.append("Key class: ").append(iterator.getKeyClass().toString());
          writer.append(" Value Class: ").append(iterator.getValueClass().toString()).append('\n');
        }
        OpenObjectIntHashMap<String> facets = null;
        if (hasOption("facets")) {
          facets = new OpenObjectIntHashMap<String>();
        }
        long count = 0;
        if (countOnly) {
          while (iterator.hasNext()) {
            Pair<?, ?> record = iterator.next();
            String key = record.getFirst().toString();
            if (facets != null) {
              facets.adjustOrPutValue(key, 1, 1); //either insert or add 1
            }
            count++;
          }
          writer.append("Count: ").append(String.valueOf(count)).append('\n');
        } else {
          long numItems = Long.MAX_VALUE;
          if (hasOption("numItems")) {
            numItems = Long.parseLong(getOption("numItems"));
            if (!hasOption("quiet")) {
              writer.append("Max Items to dump: ").append(String.valueOf(numItems)).append("\n");
            }
          }
          while (iterator.hasNext() && count < numItems) {
            Pair<?, ?> record = iterator.next();
            String key = record.getFirst().toString();
            writer.append("Key: ").append(key);
            String str = record.getSecond().toString();
            writer.append(": Value: ").append(str.length() > sub 
                                              ? str.substring(0, sub) : str);
            writer.write('\n');
            if (facets != null) {
              facets.adjustOrPutValue(key, 1, 1); //either insert or add 1
            }
            count++;
          }
          if (!hasOption("quiet")) {
            writer.append("Count: ").append(String.valueOf(count)).append('\n');
          }
        }
        if (facets != null) {
          List<String> keyList = Lists.newArrayListWithCapacity(facets.size());

          IntArrayList valueList = new IntArrayList(facets.size());
          facets.pairsSortedByKey(keyList, valueList);
          writer.append("-----Facets---\n");
          writer.append("Key\t\tCount\n");
          int i = 0;
          for (String key : keyList) {
            writer.append(key).append("\t\t").append(String.valueOf(valueList.get(i++))).append('\n');
          }
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
    new SequenceFileDumper().run(args);
  }

}
