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

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.example.kddcup.DataFileIterable;
import org.apache.mahout.cf.taste.example.kddcup.KDDCupDataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>Runs "track 2" of the KDD Cup competition using whatever recommender is inside {@link Track2Recommender}
 * and attempts to output the result in the correct contest format.</p>
 *
 * <p>Run as: {@code Track2Runner [track 2 data file directory] [output file]}</p>
 */
public final class Track2Runner {

  private static final Logger log = LoggerFactory.getLogger(Track2Runner.class);

  private Track2Runner() {
  }

  public static void main(String[] args) throws Exception {

    File dataFileDirectory = new File(args[0]);
    if (!dataFileDirectory.exists() || !dataFileDirectory.isDirectory()) {
      throw new IllegalArgumentException("Bad data file directory: " + dataFileDirectory);
    }

    long start = System.currentTimeMillis();

    KDDCupDataModel model = new KDDCupDataModel(KDDCupDataModel.getTrainingFile(dataFileDirectory));
    Track2Recommender recommender = new Track2Recommender(model, dataFileDirectory);

    long end = System.currentTimeMillis();
    log.info("Loaded model in {}s", (end - start) / 1000);
    start = end;

    Collection<Track2Callable> callables = Lists.newArrayList();
    for (Pair<PreferenceArray,long[]> tests : new DataFileIterable(KDDCupDataModel.getTestFile(dataFileDirectory))) {
      PreferenceArray userTest = tests.getFirst();
      callables.add(new Track2Callable(recommender, userTest));
    }

    int cores = Runtime.getRuntime().availableProcessors();
    log.info("Running on {} cores", cores);
    ExecutorService executor = Executors.newFixedThreadPool(cores);
    List<Future<UserResult>> futures = executor.invokeAll(callables);
    executor.shutdown();

    end = System.currentTimeMillis();
    log.info("Ran recommendations in {}s", (end - start) / 1000);
    start = end;

    OutputStream out = new BufferedOutputStream(new FileOutputStream(new File(args[1])));
    try {
      long lastUserID = Long.MIN_VALUE;
      for (Future<UserResult> future : futures) {
        UserResult result = future.get();
        long userID = result.getUserID();
        if (userID <= lastUserID) {
          throw new IllegalStateException();
        }
        lastUserID = userID;
        out.write(result.getResultBytes());
      }
    } finally {
      Closeables.close(out, false);
    }

    end = System.currentTimeMillis();
    log.info("Wrote output in {}s", (end - start) / 1000);
  }

}
