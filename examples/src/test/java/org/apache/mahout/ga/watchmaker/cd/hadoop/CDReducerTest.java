package org.apache.mahout.ga.watchmaker.cd.hadoop;

import junit.framework.TestCase;
import org.apache.hadoop.io.LongWritable;
import org.apache.mahout.ga.watchmaker.cd.CDFitness;
import org.apache.mahout.utils.DummyOutputCollector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class CDReducerTest extends TestCase {

  private final int nbevals = 100;

  private List<CDFitness> evaluations;

  private CDFitness expected;

  @Override
  protected void setUp() throws Exception {
    // generate random evaluatons and calculate expectations
    evaluations = new ArrayList<CDFitness>();
    Random rng = new Random();
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    for (int index = 0; index < nbevals; index++) {
      CDFitness fitness = new CDFitness(rng.nextInt(100), rng.nextInt(100), rng
          .nextInt(100), rng.nextInt(100));
      tp += fitness.getTp();
      fp += fitness.getFp();
      tn += fitness.getTn();
      fn += fitness.getFn();

      evaluations.add(fitness);
    }
    expected = new CDFitness(tp, fp, tn, fn);
  }

  public void testReduce() throws IOException {
    CDReducer reducer = new CDReducer();
    DummyOutputCollector<LongWritable, CDFitness> collector = new DummyOutputCollector<LongWritable, CDFitness>();
    LongWritable zero = new LongWritable(0);
    reducer.reduce(zero, evaluations.iterator(), collector, null);

    // check if the expectations are met
    Set<String> keys = collector.getKeys();
    assertEquals("nb keys", 1, keys.size());
    assertTrue("bad key", keys.contains(zero.toString()));

    assertEquals("nb values", 1, collector.getValue(zero.toString()).size());
    CDFitness fitness = collector.getValue(zero.toString()).get(0);
    assertEquals(expected, fitness);

  }

}
