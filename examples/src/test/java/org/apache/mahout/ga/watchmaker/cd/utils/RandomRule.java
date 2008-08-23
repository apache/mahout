package org.apache.mahout.ga.watchmaker.cd.utils;

import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.DataLine;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.ga.watchmaker.cd.hadoop.CDMapper;

import java.util.Random;

public class RandomRule implements Rule {

  private final Random rng;

  private final int ruleid;
  
  private final int target;

  public RandomRule(int ruleid, int target, Random rng) {
    this.ruleid = ruleid;
    this.target = target;
    this.rng = rng;
  }

  public int classify(DataLine dl) {
    int label = dl.getLabel();
    int prediction = rng.nextInt(2);

    CDFitness fitness = CDMapper.evaluate(target, prediction, label);
    RandomRuleResults.addResult(ruleid, fitness);

    return prediction;
  }
}
