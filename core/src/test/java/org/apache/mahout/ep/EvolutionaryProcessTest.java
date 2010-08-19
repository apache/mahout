package org.apache.mahout.ep;

import org.junit.Assert;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ExecutionException;

public class EvolutionaryProcessTest {
  @Test
  public void converges() throws ExecutionException, InterruptedException {
    State<Foo> s0 = new State<Foo>(new double[5], 1);
    s0.setPayload(new Foo());
    s0.setRand(new Random(1));
    EvolutionaryProcess<Foo> ep = new EvolutionaryProcess<Foo>(10, 100, s0);

    State<Foo> best = null;
    for (int i = 0; i < 10; i++) {
      best = ep.parallelDo(new EvolutionaryProcess.Function<Foo>() {
        @Override
        double apply(Foo payload, double[] params) {
          int i = 1;
          double sum = 0;
          for (double x : params) {
            sum += i * (x - i) * (x - i);
          }
          return -sum;
        }
      });

      ep.mutatePopulation(3);

      System.out.printf("%.3f\n", best.getValue());
    }

    Assert.assertNotNull(best);
    Assert.assertEquals(0, best.getValue(), 0.02);
  }

  private static class Foo implements Copyable<Foo> {
    @Override
    public Foo copy() {
      return this;
    }
  }
}
