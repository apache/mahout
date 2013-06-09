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
 * <p>Runs "track 1" of the KDD Cup competition using whatever recommender is inside {@link Track1Recommender}
 * and attempts to output the result in the correct contest format.</p>
 *
 * <p>Run as: {@code Track1Runner [track 1 data file directory] [output file]}</p>
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

    long start = System.currentTimeMillis();

    KDDCupDataModel model = new KDDCupDataModel(KDDCupDataModel.getTrainingFile(dataFileDirectory));
    Track1Recommender recommender = new Track1Recommender(model);

    long end = System.currentTimeMillis();
    log.info("Loaded model in {}s", (end - start) / 1000);
    start = end;

    Collection<Track1Callable> callables = Lists.newArrayList();
    for (Pair<PreferenceArray,long[]> tests : new DataFileIterable(KDDCupDataModel.getTestFile(dataFileDirectory))) {
      PreferenceArray userTest = tests.getFirst();
      callables.add(new Track1Callable(recommender, userTest));
    }

    int cores = Runtime.getRuntime().availableProcessors();
    log.info("Running on {} cores", cores);
    ExecutorService executor = Executors.newFixedThreadPool(cores);
    List<Future<byte[]>> results = executor.invokeAll(callables);
    executor.shutdown();

    end = System.currentTimeMillis();
    log.info("Ran recommendations in {}s", (end - start) / 1000);
    start = end;

    OutputStream out = new BufferedOutputStream(new FileOutputStream(new File(args[1])));
    try {
      for (Future<byte[]> result : results) {
        for (byte estimate : result.get()) {
          out.write(estimate);
        }
      }
    } finally {
      Closeables.close(out, false);
    }

    end = System.currentTimeMillis();
    log.info("Wrote output in {}s", (end - start) / 1000);
  }

}
