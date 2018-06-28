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

import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class Track1Callable implements Callable<byte[]> {

  private static final Logger log = LoggerFactory.getLogger(Track1Callable.class);
  private static final AtomicInteger COUNT = new AtomicInteger();

  private final Recommender recommender;
  private final PreferenceArray userTest;

  Track1Callable(Recommender recommender, PreferenceArray userTest) {
    this.recommender = recommender;
    this.userTest = userTest;
  }

  @Override
  public byte[] call() throws TasteException {
    long userID = userTest.get(0).getUserID();
    byte[] result = new byte[userTest.length()];
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
      result[i] = EstimateConverter.convert(estimate, userID, itemID);
    }

    if (COUNT.incrementAndGet() % 10000 == 0) {
      log.info("Completed {} users", COUNT.get());
    }

    return result;
  }

}
