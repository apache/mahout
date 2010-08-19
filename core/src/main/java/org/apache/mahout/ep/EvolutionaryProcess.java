package org.apache.mahout.ep;

import com.google.common.collect.Lists;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class EvolutionaryProcess<T extends Copyable<T>> {
  private ExecutorService pool;
  private List<State<T>> population;
  private int populationSize;

  public EvolutionaryProcess(int threadCount, int populationSize, State<T> seed) {
    this.populationSize = populationSize;
    pool = Executors.newFixedThreadPool(threadCount);
    population = Lists.newArrayList();
    for (int i = 0; i < populationSize; i++) {
      population.add(seed.mutate());
    }
  }

  public void mutatePopulation(int survivors) {
    Collections.sort(population);
    List<State<T>> parents = Lists.newArrayList(population.subList(0, survivors));
    population.subList(survivors, population.size()).clear();

    int i = 0;
    while (population.size() < populationSize) {
      population.add(parents.get(i % survivors).mutate());
      i++;
    }
  }

  public State<T> parallelDo(final Function<T> fn) throws InterruptedException, ExecutionException {
    Collection<Callable<State<T>>> tasks = Lists.newArrayList();
    for (final State<T> state : population) {
      tasks.add(new Callable<State<T>>() {
        @Override
        public State<T> call() throws Exception {
          double v = fn.apply(state.getPayload(), state.getMappedParams());
          state.setValue(v);
          return state;
        }
      });
    }
    List<Future<State<T>>> r = pool.invokeAll(tasks);

    double max = Double.NEGATIVE_INFINITY;
    State<T> best = null;
    for (Future<State<T>> future : r) {
      State<T> s = future.get();
      double value = s.getValue();
      if (!Double.isNaN(value) && value >= max) {
        max = value;
        best = s;
      }
    }
    if (best == null) {
      best = r.get(0).get();
    }

    return best;
  }

  public abstract static class Function<U> {
    abstract double apply(U payload, double[] params);
  }
}
