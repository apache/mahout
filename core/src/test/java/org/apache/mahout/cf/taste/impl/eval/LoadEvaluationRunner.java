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

/**
 *
 *
 **/
public class LoadEvaluationRunner {

  public static void main(String[] args) throws Exception {
    DataModel model = new FileDataModel(new File(args[0]));
    ItemSimilarity similarity = new EuclideanDistanceSimilarity(model);
    Recommender recommender = new GenericItemBasedRecommender(model, similarity);//Use an item-item recommender
    System.out.println("Run Items");
    for (int i = 0; i < 10; i++){
      LoadEvaluator.runLoad(recommender);
    }
    System.out.println("Run Users");
    UserSimilarity userSim = new EuclideanDistanceSimilarity(model);
    UserNeighborhood neighborhood = new NearestNUserNeighborhood(10, userSim, model);
    recommender = new GenericUserBasedRecommender(model, neighborhood, userSim);
    for (int i = 0; i < 10; i++){
      LoadEvaluator.runLoad(recommender);
    }

  }

}
