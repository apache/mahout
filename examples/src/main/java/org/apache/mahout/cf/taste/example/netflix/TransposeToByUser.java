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

package org.apache.mahout.cf.taste.example.netflix;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.commons.cli2.OptionException;
import org.apache.mahout.cf.taste.example.TasteOptionParser;
import org.apache.mahout.cf.taste.impl.common.FastMap;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class TransposeToByUser {
  
  private static final Logger log = LoggerFactory.getLogger(TransposeToByUser.class);
  
  private TransposeToByUser() { }
  
  public static void main(String[] args) throws IOException, OptionException {
    
    File dataDirectory = TasteOptionParser.getRatings(args);
    File byItemDirectory = new File(dataDirectory, "training_set");
    File byUserDirectory = new File(dataDirectory, "training_set_by_user");

    Preconditions.checkArgument(dataDirectory.exists() && dataDirectory.isDirectory(),
                                "%s is not a directory", dataDirectory);
    Preconditions.checkArgument(byItemDirectory.exists() && byItemDirectory.isDirectory(),
                                "%s is not a directory", byItemDirectory);
    Preconditions.checkArgument(!byUserDirectory.exists(), "%s already exists", byUserDirectory);
    
    byUserDirectory.mkdirs();
    
    Map<String, List<String>> byUserEntryCache = new FastMap<String, List<String>>(100000);
    
    for (File byItemFile : byItemDirectory.listFiles()) {
      log.info("Processing {}", byItemFile);
      Iterator<String> lineIterator = new FileLineIterable(byItemFile, false).iterator();
      String line = lineIterator.next();
      String movieIDString = line.substring(0, line.length() - 1);
      while (lineIterator.hasNext()) {
        line = lineIterator.next();
        int firstComma = line.indexOf(',');
        String userIDString = line.substring(0, firstComma);
        int secondComma = line.indexOf(',', firstComma + 1);
        String ratingString = line.substring(firstComma, secondComma); // keep comma
        List<String> cachedLines = byUserEntryCache.get(userIDString);
        if (cachedLines == null) {
          cachedLines = Lists.newArrayList();
          byUserEntryCache.put(userIDString, cachedLines);
        }
        cachedLines.add(movieIDString + ratingString);
        maybeFlushCache(byUserDirectory, byUserEntryCache);
      }
      
    }
    
  }
  
  private static void maybeFlushCache(File byUserDirectory,
                                      Map<String, List<String>> byUserEntryCache) throws IOException {
    if (byUserEntryCache.size() >= 100000) {
      log.info("Flushing cache");
      for (Map.Entry<String, List<String>> entry : byUserEntryCache.entrySet()) {
        String userID = entry.getKey();
        List<String> lines = entry.getValue();
        int userIDValue = Integer.parseInt(userID);
        File intermediateDir = new File(byUserDirectory, String.valueOf(userIDValue % 10000));
        intermediateDir.mkdirs();
        File userIDFile = new File(intermediateDir, userIDValue / 10000 + ".txt");
        appendStringsToFile(lines, userIDFile);
      }
      byUserEntryCache.clear();
    }
  }
  
  private static void appendStringsToFile(Iterable<String> strings, File file) throws IOException {
    PrintWriter outputStreamWriter =
      new PrintWriter(new OutputStreamWriter(new FileOutputStream(file, true), Charsets.UTF_8));
    try {
      for (String s : strings) {
        outputStreamWriter.println(s);
      }
    } finally {
      Closeables.closeQuietly(outputStreamWriter);
    }
  }
  
}
