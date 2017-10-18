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

package org.apache.mahout.cf.taste.example.email;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

import java.io.IOException;
import java.util.regex.Pattern;

public final class EmailUtility {

  public static final String SEPARATOR = "separator";
  public static final String MSG_IDS_PREFIX = "msgIdsPrefix";
  public static final String FROM_PREFIX = "fromPrefix";
  public static final String MSG_ID_DIMENSION = "msgIdDim";
  public static final String FROM_INDEX = "fromIdx";
  public static final String REFS_INDEX = "refsIdx";
  private static final String[] EMPTY = new String[0];
  private static final Pattern ADDRESS_CLEANUP = Pattern.compile("mailto:|<|>|\\[|\\]|\\=20");
  private static final Pattern ANGLE_BRACES = Pattern.compile("<|>");
  private static final Pattern SPACE_OR_CLOSE_ANGLE = Pattern.compile(">|\\s+");
  public static final Pattern WHITESPACE = Pattern.compile("\\s*");

  private EmailUtility() {
  }

  /**
   * Strip off some spurious characters that make it harder to dedup
   */
  public static String cleanUpEmailAddress(CharSequence address) {
    //do some cleanup to normalize some things, like: Key: karthik ananth <karthik.jcecs@gmail.com>: Value: 178
    //Key: karthik ananth [mailto:karthik.jcecs@gmail.com]=20: Value: 179
    //TODO: is there more to clean up here?
    return ADDRESS_CLEANUP.matcher(address).replaceAll("");
  }

  public static void loadDictionaries(Configuration conf, String fromPrefix,
                                      OpenObjectIntHashMap<String> fromDictionary,
                                      String msgIdPrefix,
                                      OpenObjectIntHashMap<String> msgIdDictionary) throws IOException {

    Path[] localFiles = HadoopUtil.getCachedFiles(conf);
    FileSystem fs = FileSystem.getLocal(conf);
    for (Path dictionaryFile : localFiles) {

      // key is word value is id

      OpenObjectIntHashMap<String> dictionary = null;
      if (dictionaryFile.getName().startsWith(fromPrefix)) {
        dictionary = fromDictionary;
      } else if (dictionaryFile.getName().startsWith(msgIdPrefix)) {
        dictionary = msgIdDictionary;
      }
      if (dictionary != null) {
        dictionaryFile = fs.makeQualified(dictionaryFile);
        for (Pair<Writable, IntWritable> record
            : new SequenceFileIterable<Writable, IntWritable>(dictionaryFile, true, conf)) {
          dictionary.put(record.getFirst().toString(), record.getSecond().get());
        }
      }
    }

  }

  public static String[] parseReferences(CharSequence rawRefs) {
    String[] splits;
    if (rawRefs != null && rawRefs.length() > 0) {
      splits = SPACE_OR_CLOSE_ANGLE.split(rawRefs);
      for (int i = 0; i < splits.length; i++) {
        splits[i] = ANGLE_BRACES.matcher(splits[i]).replaceAll("");
      }
    } else {
      splits = EMPTY;
    }
    return splits;
  }

  public enum Counters {
    NO_MESSAGE_ID, NO_FROM_ADDRESS
  }
}
