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

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

final class Track2Callable implements Callable<UserResult> {

  private static final Logger log = LoggerFactory.getLogger(Track2Callable.class);
  private static final AtomicInteger COUNT = new AtomicInteger();

  private final Recommender recommender;
  private final PreferenceArray userTest;

  Track2Callable(Recommender recommender, PreferenceArray userTest) {
    this.recommender = recommender;
    this.userTest = userTest;
  }

  @Override
  public UserResult call() throws TasteException {

    int testSize = userTest.length();
    if (testSize != 6) {
      throw new IllegalArgumentException("Expecting 6 items for user but got " + userTest);
    }
    long userID = userTest.get(0).getUserID();
    TreeMap<Double,Long> estimateToItemID = new TreeMap<Double,Long>(Collections.reverseOrder());

    for (int i = 0; i < testSize; i++) {
      long itemID = userTest.getItemID(i);
      double estimate;
      try {
        estimate = recommender.estimatePreference(userID, itemID);
      } catch (NoSuchItemException nsie) {
        // OK in the sample data provided before the contest, should never happen otherwise
        log.warn("Unknown item {}; OK unless this is the real contest data", itemID);
        continue;
      }

      if (!Double.isNaN(estimate)) {
        estimateToItemID.put(estimate, itemID);
      }
    }

    Collection<Long> itemIDs = estimateToItemID.values();
    List<Long> topThree = Lists.newArrayList(itemIDs);
    if (topThree.size() > 3) {
      topThree = topThree.subList(0, 3);
    } else if (topThree.size() < 3) {
      log.warn("Unable to recommend three items for {}", userID);
      // Some NaNs - just guess at the rest then
      Collection<Long> newItemIDs = Sets.newHashSetWithExpectedSize(3);
      newItemIDs.addAll(itemIDs);
      int i = 0;
      while (i < testSize && newItemIDs.size() < 3) {
        newItemIDs.add(userTest.getItemID(i));
        i++;
      }
      topThree = Lists.newArrayList(newItemIDs);
    }
    if (topThree.size() != 3) {
      throw new IllegalStateException();
    }

    boolean[] result = new boolean[testSize];
    for (int i = 0; i < testSize; i++) {
      result[i] = topThree.contains(userTest.getItemID(i));
    }

    if (COUNT.incrementAndGet() % 1000 == 0) {
      log.info("Completed {} users", COUNT.get());
    }

    return new UserResult(userID, result);
  }
}
