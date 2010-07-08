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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.IOUtils;

/**
 * some helper methods for the hadoop-related stuff in org.apache.mahout.cf.taste
 */
public final class TasteHadoopUtils {

  /** standard delimiter of textual preference data */
  private static final Pattern PREFERENCE_TOKEN_DELIMITER = Pattern.compile("[\t,]");

  private TasteHadoopUtils() {
  }

  /**
   * splits a preference data line into string tokens
   *
   * @param line
   * @return
   */
  public static String[] splitPrefTokens(String line) {
    return PREFERENCE_TOKEN_DELIMITER.split(line);
  }

  /** a path filter used to read files written by hadoop */
  public static final PathFilter PARTS_FILTER = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      return path.getName().startsWith("part-");
    }
  };

  /**
   * maps a long to an int
   *
   * @param id
   * @return
   */
  public static int idToIndex(long id) {
    return 0x7FFFFFFF & ((int) id ^ (int) (id >>> 32));
  }
  
  /**
   * reads a text-based outputfile that only contains an int
   * 
   * @param conf
   * @param outputDir
   * @return
   * @throws IOException
   */
  public static int readIntFromFile(Configuration conf, Path outputDir) throws IOException {
    FileSystem fs = FileSystem.get(conf);
    Path outputFile = fs.listStatus(outputDir, TasteHadoopUtils.PARTS_FILTER)[0].getPath();
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
