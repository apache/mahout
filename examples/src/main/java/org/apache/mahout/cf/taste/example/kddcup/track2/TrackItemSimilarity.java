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

import java.io.File;
import java.io.IOException;
import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.example.kddcup.KDDCupDataModel;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.common.iterator.FileLineIterable;

final class TrackItemSimilarity implements ItemSimilarity {

  private final FastByIDMap<TrackData> trackData;

  TrackItemSimilarity(File dataFileDirectory) throws IOException {
    trackData = new FastByIDMap<TrackData>();
    for (String line : new FileLineIterable(KDDCupDataModel.getTrackFile(dataFileDirectory))) {
      TrackData trackDatum = new TrackData(line);
      trackData.put(trackDatum.getTrackID(), trackDatum);
    }
  }

  @Override
  public double itemSimilarity(long itemID1, long itemID2) {
    if (itemID1 == itemID2) {
      return 1.0;
    }
    TrackData data1 = trackData.get(itemID1);
    TrackData data2 = trackData.get(itemID2);
    if (data1 == null || data2 == null) {
      return 0.0;
    }

    // Arbitrarily decide that same album means "very similar"
    if (data1.getAlbumID() != TrackData.NO_VALUE_ID && data1.getAlbumID() == data2.getAlbumID()) {
      return 0.9;
    }
    // ... and same artist means "fairly similar"
    if (data1.getArtistID() != TrackData.NO_VALUE_ID && data1.getArtistID() == data2.getArtistID()) {
      return 0.7;
    }

    // Tanimoto coefficient similarity based on genre, but maximum value of 0.25
    FastIDSet genres1 = data1.getGenreIDs();
    FastIDSet genres2 = data2.getGenreIDs();
    if (genres1 == null || genres2 == null) {
      return 0.0;
    }
    int intersectionSize = genres1.intersectionSize(genres2);
    if (intersectionSize == 0) {
      return 0.0;
    }
    int unionSize = genres1.size() + genres2.size() - intersectionSize;
    return intersectionSize / (4.0 * unionSize);
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) {
    int length = itemID2s.length;
    double[] result = new double[length];
    for (int i = 0; i < length; i++) {
      result[i] = itemSimilarity(itemID1, itemID2s[i]);
    }
    return result;
  }

  @Override
  public long[] allSimilarItemIDs(long itemID) {
    FastIDSet allSimilarItemIDs = new FastIDSet();
    LongPrimitiveIterator allItemIDs = trackData.keySetIterator();
    while (allItemIDs.hasNext()) {
      long possiblySimilarItemID = allItemIDs.nextLong();
      if (!Double.isNaN(itemSimilarity(itemID, possiblySimilarItemID))) {
        allSimilarItemIDs.add(possiblySimilarItemID);
      }
    }
    return allSimilarItemIDs.toArray();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // do nothing
  }

}
