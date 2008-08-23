package org.apache.mahout.ga.watchmaker.cd.hadoop;

import junit.framework.TestCase;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.ga.watchmaker.cd.utils.RandomRule;
import org.apache.mahout.ga.watchmaker.cd.utils.RandomRuleResults;
import org.uncommons.maths.random.MersenneTwisterRNG;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CDMahoutEvaluatorTest extends TestCase {

  public void testEvaluate() throws Exception {
    int nbrules = 100;
    Random rng = new MersenneTwisterRNG();
    int target = 1;

    // random rules
    List<Rule> rules = new ArrayList<Rule>();
    for (int index = 0; index < nbrules; index++) {
      rules.add(new RandomRule(index, target, rng));
    }

    // dataset
    Path input = new Path("build/test-classes/wdbc");
    CDMahoutEvaluator.InitializeDataSet(input);

    // evaluate the rules
    List<CDFitness> results = new ArrayList<CDFitness>();
    CDMahoutEvaluator.evaluate(rules, target, input, results);

    // check the results
    for (int index = 0; index < nbrules; index++) {
      assertEquals("rule " + index, RandomRuleResults.getResult(index),
          results.get(index));
    }

  }

}
