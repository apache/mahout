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

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.MapFile;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents a series of {@link MapFile}s, from which one might want to look up values based on keys. It just
 * provides a simplified way to open them all up, and search from all of them.
 */
@SuppressWarnings("unchecked")
public final class MapFilesMap<K extends WritableComparable,V extends Writable> implements Closeable {

  private static final Logger log = LoggerFactory.getLogger(MapFilesMap.class);
  
  private static final PathFilter PARTS_FILTER = new PathFilter() {
    @Override
    public boolean accept(Path path) {
      return path.getName().startsWith("part-");
    }
  };
  
  private final List<MapFile.Reader> readers;
  
  public MapFilesMap(FileSystem fs, Path parentDir, Configuration conf) throws IOException {
    log.info("Creating MapFileMap from parent directory {}", parentDir);
    readers = new ArrayList<MapFile.Reader>();
    try {
      for (FileStatus status : fs.listStatus(parentDir, PARTS_FILTER)) {
        String path = status.getPath().toString();
        log.info("Adding MapFile.Reader at {}", path);
        readers.add(new MapFile.Reader(fs, path, conf));
      }
    } catch (IOException ioe) {
      close();
      throw ioe;
    }
    if (readers.isEmpty()) {
      throw new IllegalArgumentException("No MapFiles found in " + parentDir);
    }
  }
  
  /**
   * @return value reference if key is found, filled in with value data, or null if not found
   */
  public V get(K key, V value) throws IOException {
    for (MapFile.Reader reader : readers) {
      V theValue;
      if ((theValue = (V) reader.get(key, value)) != null) {
        return theValue;
      }
    }
    log.debug("No value for key {}", key);
    return null;
  }
  
  @Override
  public void close() {
    for (MapFile.Reader reader : readers) {
      try {
        reader.close();
      } catch (IOException ioe) {
        // continue
      }
    }
  }
}
