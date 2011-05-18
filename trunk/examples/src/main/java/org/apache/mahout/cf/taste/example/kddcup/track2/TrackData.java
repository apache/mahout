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

package org.apache.mahout.cf.taste.example.kddcup.track2;

import java.util.regex.Pattern;

import org.apache.mahout.cf.taste.impl.common.FastIDSet;

final class TrackData {

  private static final Pattern PIPE = Pattern.compile("\\|");
  private static final String NO_VALUE = "None";
  static final long NO_VALUE_ID = Long.MIN_VALUE;
  private static final FastIDSet NO_GENRES = new FastIDSet();

  private final long trackID;
  private final long albumID;
  private final long artistID;
  private final FastIDSet genreIDs;

  TrackData(CharSequence line) {
    String[] tokens = PIPE.split(line);
    trackID = Long.parseLong(tokens[0]);
    albumID = parse(tokens[1]);
    artistID = parse(tokens[2]);
    if (tokens.length > 3) {
      genreIDs = new FastIDSet(tokens.length - 3);
      for (int i = 3; i < tokens.length; i++) {
        genreIDs.add(Long.parseLong(tokens[i]));
      }
    } else {
      genreIDs = NO_GENRES;
    }
  }

  private static long parse(String value) {
    return NO_VALUE.equals(value) ? NO_VALUE_ID : Long.parseLong(value);
  }

  public long getTrackID() {
    return trackID;
  }

  public long getAlbumID() {
    return albumID;
  }

  public long getArtistID() {
    return artistID;
  }

  public FastIDSet getGenreIDs() {
    return genreIDs;
  }

}
