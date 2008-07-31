package org.apache.mahout.ga.watchmaker.cd.hadoop;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import junit.framework.TestCase;

import org.apache.hadoop.fs.Path;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.ga.watchmaker.cd.Rule;
import org.apache.mahout.ga.watchmaker.cd.hadoop.CDMahoutEvaluator;
import org.apache.mahout.ga.watchmaker.cd.utils.RandomRule;
import org.apache.mahout.ga.watchmaker.cd.utils.RandomRuleResults;
import org.uncommons.maths.random.MersenneTwisterRNG;

public class CDMahoutEvaluatorTest extends TestCase {

  public void testEvaluate() throws Exception {
    int nbrules = 100;
    Random rng = new MersenneTwisterRNG();

    // random rules
    List<Rule> rules = new ArrayList<Rule>();
    for (int index = 0; index < nbrules; index++) {
      rules.add(new RandomRule(index, rng));
    }

    // dataset
    Path input = new Path("build/examples-test-classes/wdbc");
    CDMahoutEvaluator.InitializeDataSet(input);

    // evaluate the rules
    List<CDFitness> results = new ArrayList<CDFitness>();
    CDMahoutEvaluator.evaluate(rules, input, results);

    // check the results
    for (int index = 0; index < nbrules; index++) {
      assertEquals("rule " + index, RandomRuleResults.getResult(index),
          results.get(index));
    }

  }

}
