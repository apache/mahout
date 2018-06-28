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

package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;

public final class LoadEvaluationRunner {

  private static final int LOOPS = 10;

  private LoadEvaluationRunner() {
  }

  public static void main(String[] args) throws Exception {

    DataModel model = new FileDataModel(new File(args[0]));

    int howMany = 10;
    if (args.length > 1) {
      howMany = Integer.parseInt(args[1]);
    }

    System.out.println("Run Items");
    ItemSimilarity similarity = new EuclideanDistanceSimilarity(model);
    Recommender recommender = new GenericItemBasedRecommender(model, similarity); // Use an item-item recommender
    for (int i = 0; i < LOOPS; i++) {
      LoadStatistics loadStats = LoadEvaluator.runLoad(recommender, howMany);
      System.out.println(loadStats);
    }

    System.out.println("Run Users");
    UserSimilarity userSim = new EuclideanDistanceSimilarity(model);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, userSim, model);
    recommender = new GenericUserBasedRecommender(model, neighborhood, userSim);
    for (int i = 0; i < LOOPS; i++) {
      LoadStatistics loadStats = LoadEvaluator.runLoad(recommender, howMany);
      System.out.println(loadStats);
    }

  }

}
