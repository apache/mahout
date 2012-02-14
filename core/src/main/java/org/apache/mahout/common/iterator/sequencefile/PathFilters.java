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

package org.apache.mahout.common.iterator.sequencefile;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;

/**
 * Supplies some useful and repeatedly-used instances of {@link PathFilter}.
 */
public final class PathFilters {

  private static final PathFilter PART_FILE_INSTANCE = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      String name = path.getName();
      return name.startsWith("part-") && !name.endsWith(".crc");
    }
  };
  
  /**
   * Pathfilter to read the final clustering file.
   */
  private static final PathFilter CLUSTER_FINAL = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      String name = path.getName();
      return name.startsWith("clusters-") && name.endsWith("-final");
    }
  };

  private static final PathFilter LOGS_CRC_INSTANCE = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      String name = path.getName();
      return !(name.endsWith(".crc") || name.startsWith(".") || name.startsWith("_"));
    }
  };

  private PathFilters() {
  }

  /**
   * @return {@link PathFilter} that accepts paths whose file name starts with "part-". Excludes
   * ".crc" files.
   */
  public static PathFilter partFilter() {
    return PART_FILE_INSTANCE;
  }
  
  /**
   * @return {@link PathFilter} that accepts paths whose file name starts with "part-" and ends with "-final".
   */
  public static PathFilter finalPartFilter() {
    return CLUSTER_FINAL;
  }

  /**
   * @return {@link PathFilter} that rejects paths whose file name starts with "_" (e.g. Cloudera
   * _SUCCESS files or Hadoop _logs), or "." (e.g. local hidden files), or ends with ".crc"
   */
  public static PathFilter logsCRCFilter() {
    return LOGS_CRC_INSTANCE;
  }

}
