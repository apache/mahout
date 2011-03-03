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

package org.apache.mahout.cf.taste.example.kddcup.track1;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.example.kddcup.DataFileIterable;
import org.apache.mahout.cf.taste.example.kddcup.KDDCupDataModel;
import org.apache.mahout.cf.taste.example.kddcup.KDDCupRecommender;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>Runs "track 1" of the KDD Cup competition using whatever recommender is inside {@link KDDCupRecommender}
 * and attempts to output the result in the correct contest format.</p>
 *
 * <p>Run as: <code>Track1Runner [track 1 data file directory] [output file]</code></p>
 */
public final class Track1Runner {

  private static final Logger log = LoggerFactory.getLogger(Track1Runner.class);

  private Track1Runner() {
  }

  public static void main(String[] args) throws Exception {

    File dataFileDirectory = new File(args[0]);
    if (!dataFileDirectory.exists() || !dataFileDirectory.isDirectory()) {
      throw new IllegalArgumentException("Bad data file directory: " + dataFileDirectory);
    }

    KDDCupDataModel model = new KDDCupDataModel(KDDCupDataModel.getTrainingFile(dataFileDirectory));
    KDDCupRecommender recommender = new KDDCupRecommender(model);

    File outFile = new File(args[1]);
    OutputStream out = new FileOutputStream(outFile);

    for (Pair<PreferenceArray,long[]> tests : new DataFileIterable(KDDCupDataModel.getTestFile(dataFileDirectory))) {

      PreferenceArray userTest = tests.getFirst();
      long userID = userTest.get(0).getUserID();
      for (int i = 0; i < userTest.length(); i++) {
        long itemID = userTest.getItemID(i);
        double estimate;
        try {
          estimate = recommender.estimatePreference(userID, itemID);
        } catch (NoSuchItemException nsie) {
          // OK in the sample data provided before the contest, should never happen otherwise
          log.warn("Unknown item {}; OK unless this is the real contest data", itemID);
          continue;
        }

        log.info("Estimate for user {}, item {}: ", new Object[] {userID, itemID, estimate});

        int scaledEstimate = (int) ((estimate / 100.0) * 255.0);
        if (scaledEstimate > 255) {
          scaledEstimate = 255;
        } else if (scaledEstimate < 0) {
          scaledEstimate = 0;
        }

        out.write(scaledEstimate);

      }

    }

    out.close();

  }


}
