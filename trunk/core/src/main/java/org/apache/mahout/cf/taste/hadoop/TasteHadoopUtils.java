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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.SequenceFile;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.map.OpenIntLongHashMap;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.regex.Pattern;

/**
 * Some helper methods for the hadoop-related stuff in org.apache.mahout.cf.taste
 */
public final class TasteHadoopUtils {

  /** Standard delimiter of textual preference data */
  private static final Pattern PREFERENCE_TOKEN_DELIMITER = Pattern.compile("[\t,]");

  private TasteHadoopUtils() {
  }

  /**
   * Splits a preference data line into string tokens
   */
  public static String[] splitPrefTokens(CharSequence line) {
    return PREFERENCE_TOKEN_DELIMITER.split(line);
  }

  /**
   * A path filter used to read files written by Hadoop.
   */
  public static final PathFilter PARTS_FILTER = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      return path.getName().startsWith("part-");
    }
  };

  /**
   * Maps a long to an int
   */
  public static int idToIndex(long id) {
    return 0x7FFFFFFF & ((int) id ^ (int) (id >>> 32));
  }

  /**
   * Reads a binary mapping file
   */
  public static OpenIntLongHashMap readItemIDIndexMap(String itemIDIndexPathStr, Configuration conf) {
    OpenIntLongHashMap indexItemIDMap = new OpenIntLongHashMap();
    try {
      Path unqualifiedItemIDIndexPath = new Path(itemIDIndexPathStr);
      FileSystem fs = FileSystem.get(unqualifiedItemIDIndexPath.toUri(), conf);
      Path itemIDIndexPath = new Path(itemIDIndexPathStr).makeQualified(fs);

      VarIntWritable index = new VarIntWritable();
      VarLongWritable id = new VarLongWritable();
      for (FileStatus status : fs.listStatus(itemIDIndexPath, PARTS_FILTER)) {
        String path = status.getPath().toString();
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(path).makeQualified(fs), conf);
        while (reader.next(index, id)) {
          indexItemIDMap.put(index.get(), id.get());
        }
        reader.close();
      }
    } catch (IOException ioe) {
      throw new IllegalStateException(ioe);
    }
    return indexItemIDMap;
  }

  /**
   * Reads a text-based outputfile that only contains an int
   */
  public static int readIntFromFile(Configuration conf, Path outputDir) throws IOException {
    FileSystem fs = FileSystem.get(outputDir.toUri(), conf);
    Path outputFile = fs.listStatus(outputDir, PARTS_FILTER)[0].getPath();
    InputStream in = null;
    try  {
      in = fs.open(outputFile);
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      IOUtils.copyBytes(in, out, conf);
      return Integer.parseInt(new String(out.toByteArray(), Charset.forName("UTF-8")).trim());
    } finally {
      IOUtils.closeStream(in);
    }
  }
}
