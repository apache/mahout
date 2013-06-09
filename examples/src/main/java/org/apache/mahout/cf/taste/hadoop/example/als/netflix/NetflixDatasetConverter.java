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

package org.apache.mahout.cf.taste.hadoop.example.als.netflix;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.regex.Pattern;

import com.google.common.base.Charsets;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.common.iterator.FileLineIterable;
import org.apache.mahout.common.iterator.FileLineIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** converts the raw files provided by netflix to an appropriate input format */
public final class NetflixDatasetConverter {

  private static final Logger log = LoggerFactory.getLogger(NetflixDatasetConverter.class);

  private static final Pattern SEPARATOR = Pattern.compile(",");
  private static final String MOVIE_DENOTER = ":";
  private static final String TAB = "\t";
  private static final String NEWLINE = "\n";

  private NetflixDatasetConverter() {
  }

  public static void main(String[] args) throws IOException {

    if (args.length != 4) {
      System.err.println("Usage: NetflixDatasetConverter /path/to/training_set/ /path/to/qualifying.txt "
          + "/path/to/judging.txt /path/to/destination");
      return;
    }

    String trainingDataDir = args[0];
    String qualifyingTxt = args[1];
    String judgingTxt = args[2];
    Path outputPath = new Path(args[3]);

    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(outputPath.toUri(), conf);

    log.info("Creating training set at {}/trainingSet/ratings.tsv ...", outputPath);
    BufferedWriter writer  = null;
    try {
      FSDataOutputStream outputStream = fs.create(new Path(outputPath, "trainingSet/ratings.tsv"));
      writer = new BufferedWriter(new OutputStreamWriter(outputStream, Charsets.UTF_8));

      int ratingsProcessed = 0;
      for (File movieRatings : new File(trainingDataDir).listFiles()) {
        FileLineIterator lines = null;
        try  {
          lines = new FileLineIterator(movieRatings);
          boolean firstLineRead = false;
          String movieID = null;
          while (lines.hasNext()) {
            String line = lines.next();
            if (firstLineRead) {
              String[] tokens = SEPARATOR.split(line);
              String userID = tokens[0];
              String rating = tokens[1];
              writer.write(userID + TAB + movieID + TAB + rating + NEWLINE);
              ratingsProcessed++;
              if (ratingsProcessed % 1000000 == 0) {
                log.info("{} ratings processed...", ratingsProcessed);
              }
            } else {
              movieID = line.replaceAll(MOVIE_DENOTER, "");
              firstLineRead = true;
            }
          }
        } finally {
          Closeables.close(lines, true);
        }
      }
      log.info("{} ratings processed. done.", ratingsProcessed);
    } finally {
      Closeables.close(writer, false);
    }

    log.info("Reading probes...");
    List<Preference> probes = Lists.newArrayListWithExpectedSize(2817131);
    long currentMovieID = -1;
    for (String line : new FileLineIterable(new File(qualifyingTxt))) {
      if (line.contains(MOVIE_DENOTER)) {
        currentMovieID = Long.parseLong(line.replaceAll(MOVIE_DENOTER, ""));
      } else {
        long userID = Long.parseLong(SEPARATOR.split(line)[0]);
        probes.add(new GenericPreference(userID, currentMovieID, 0));
      }
    }
    log.info("{} probes read...", probes.size());

    log.info("Reading ratings, creating probe set at {}/probeSet/ratings.tsv ...", outputPath);
    writer = null;
    try {
      FSDataOutputStream outputStream = fs.create(new Path(outputPath, "probeSet/ratings.tsv"));
      writer = new BufferedWriter(new OutputStreamWriter(outputStream, Charsets.UTF_8));

      int ratingsProcessed = 0;
      for (String line : new FileLineIterable(new File(judgingTxt))) {
        if (line.contains(MOVIE_DENOTER)) {
          currentMovieID = Long.parseLong(line.replaceAll(MOVIE_DENOTER, ""));
        } else {
          float rating = Float.parseFloat(SEPARATOR.split(line)[0]);
          Preference pref = probes.get(ratingsProcessed);
          Preconditions.checkState(pref.getItemID() == currentMovieID);
          ratingsProcessed++;
          writer.write(pref.getUserID() + TAB + pref.getItemID() + TAB + rating + NEWLINE);
          if (ratingsProcessed % 1000000 == 0) {
            log.info("{} ratings processed...", ratingsProcessed);
          }
        }
      }
      log.info("{} ratings processed. done.", ratingsProcessed);
    } finally {
      Closeables.close(writer, false);
    }
  }


}
