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

package org.apache.mahout.cf.taste.hadoop;

import com.google.common.primitives.Longs;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.map.OpenIntLongHashMap;

import java.util.regex.Pattern;

/**
 * Some helper methods for the hadoop-related stuff in org.apache.mahout.cf.taste
 */
public final class TasteHadoopUtils {

  public static final int USER_ID_POS = 0;
  public static final int ITEM_ID_POS = 1;

  /** Standard delimiter of textual preference data */
  private static final Pattern PREFERENCE_TOKEN_DELIMITER = Pattern.compile("[\t,]");

  private TasteHadoopUtils() {}

  /**
   * Splits a preference data line into string tokens
   */
  public static String[] splitPrefTokens(CharSequence line) {
    return PREFERENCE_TOKEN_DELIMITER.split(line);
  }

  /**
   * Maps a long to an int with range of 0 to Integer.MAX_VALUE-1
   */
  public static int idToIndex(long id) {
    return 0x7FFFFFFF & Longs.hashCode(id) % 0x7FFFFFFE;
  }

  public static int readID(String token, boolean usesLongIDs) {
    return usesLongIDs ? idToIndex(Long.parseLong(token)) : Integer.parseInt(token);
  }

  /**
   * Reads a binary mapping file
   */
  public static OpenIntLongHashMap readIDIndexMap(String idIndexPathStr, Configuration conf) {
    OpenIntLongHashMap indexIDMap = new OpenIntLongHashMap();
    Path itemIDIndexPath = new Path(idIndexPathStr);
    for (Pair<VarIntWritable,VarLongWritable> record
         : new SequenceFileDirIterable<VarIntWritable,VarLongWritable>(itemIDIndexPath,
                                                                       PathType.LIST,
                                                                       PathFilters.partFilter(),
                                                                       null,
                                                                       true,
                                                                       conf)) {
      indexIDMap.put(record.getFirst().get(), record.getSecond().get());
    }
    return indexIDMap;
  }



}
