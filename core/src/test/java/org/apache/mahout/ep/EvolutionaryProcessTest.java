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
    for (int i = 0; i < 20  ; i++) {
      best = ep.parallelDo(new EvolutionaryProcess.Function<Foo>() {
        @Override
        public double apply(Foo payload, double[] params) {
          int i = 1;
          double sum = 0;
          for (double x : params) {
            sum += i * (x - i) * (x - i);
            i++;
          }
          return -sum;
        }
      });

      ep.mutatePopulation(3);

      System.out.printf("%10.3f %.3f\n", best.getValue(), best.getOmni());
    }

    Assert.assertNotNull(best);
    Assert.assertEquals(0, best.getValue(), 0.02);
  }

  private static class Foo implements Payload<Foo> {
    @Override
    public Foo copy() {
      return this;
    }

    @Override
    public void update(double[] params) {
      // ignore
    }
  }
}
