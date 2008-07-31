package org.apache.mahout.ga.watchmaker.cd.utils;

import java.util.Random;

import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.DataLine;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.ga.watchmaker.cd.hadoop.CDMapper;

public class RandomRule implements Rule {

  private Random rng;

  private int ruleid;

  public RandomRule(int ruleid, Random rng) {
    this.ruleid = ruleid;
    this.rng = rng;
  }

  public int classify(DataLine dl) {
    int label = dl.getLabel();
    int prediction = rng.nextInt(2);

    CDFitness fitness = CDMapper.evaluate(prediction, label);
    RandomRuleResults.addResult(ruleid, fitness);

    return prediction;
  }
}
