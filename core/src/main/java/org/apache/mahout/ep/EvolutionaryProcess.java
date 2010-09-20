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

/**
 * Allows evolutionary optimization where the state function can't be easily
 * packaged for the optimizer to execute.  A good example of this is with
 * on-line learning where optimizing the learning parameters is desirable.
 * We would like to pass training examples to the learning algorithms, but
 * we definitely want to do the training in multiple threads and then after
 * several training steps, we want to do a selection and mutation step.
 *
 * In such a case, it is highly desirable to leave most of the control flow
 * in the hands of our caller.  As such, this class provides three functions,
 * <ul>
 * <li> Storage of the evolutionary state.  The state variables have payloads
 * which can be anything that implements Payload.
 * <li> Threaded execution of a single operation on each of the members of the
 * population being evolved.  In the on-line learning example, this is used for
 * training all of the classifiers in the population.
 * <li> Propagating mutations of the most successful members of the population.
 * This propagation involves copying the state and the payload and then updating
 * the payload after mutation of the evolutionary state.
 * </ul>
 *
 * The State class that we use for storing the state of each member of the
 * population also provides parameter mapping.  Check out Mapping and State
 * for more info.
 *
 * @see Mapping
 * @see Payload
 * @see State
 *
 * @param <T> The payload class.
 */
public class EvolutionaryProcess<T extends Payload<T>> {
  // used to execute operations on the population in thread parallel.
  private transient ExecutorService pool;

  // threadCount is serialized so that we can reconstruct the thread pool
  private int threadCount;

  // list of members of the population
  private List<State<T>> population;

  // how big should the population be.  If this is changed, it will take effect
  // the next time the population is mutated.

  private int populationSize;

  public EvolutionaryProcess() {
    population = Lists.newArrayList();
  }

  /**
   * Creates an evolutionary optimization framework with specified threadiness,
   * population size and initial state.
   * @param threadCount               How many threads to use in parallelDo
   * @param populationSize            How large a population to use
   * @param seed                      An initial population member
   */
  public EvolutionaryProcess(int threadCount, int populationSize, State<T> seed) {
    this.populationSize = populationSize;
    setThreadCount(threadCount);
    initializePopulation(populationSize, seed);
  }

  private void initializePopulation(int populationSize, State<T> seed) {
    population = Lists.newArrayList(seed);
    for (int i = 0; i < populationSize; i++) {
      population.add(seed.mutate());
    }
  }

  public void add(State<T> value) {
    population.add(value);
  }

  /**
   * Nuke all but a few of the current population and then repopulate with
   * variants of the survivors.
   * @param survivors          How many survivors we want to keep.
   */
  public void mutatePopulation(int survivors) {
    // largest value first, oldest first in case of ties
    Collections.sort(population);

    // we copy here to avoid concurrent modification
    List<State<T>> parents = Lists.newArrayList(population.subList(0, survivors));
    population.subList(survivors, population.size()).clear();

    // fill out the population with offspring from the survivors
    int i = 0;
    while (population.size() < populationSize) {
      population.add(parents.get(i % survivors).mutate());
      i++;
    }
  }

  /**
   * Execute an operation on all of the members of the population with many threads.  The
   * return value is taken as the current fitness of the corresponding member.
   * @param fn    What to do on each member.  Gets payload and the mapped parameters as args.
   * @return      The member of the population with the best fitness.
   * @throws InterruptedException      Shouldn't happen.
   * @throws ExecutionException        If fn throws an exception, that exception will be collected
   * and rethrown nested in an ExecutionException.
   */
  public State<T> parallelDo(final Function fn) throws InterruptedException, ExecutionException {
    Collection<Callable<State<T>>> tasks = Lists.newArrayList();
    for (final State<T> state : population) {
      tasks.add(new Callable<State<T>>() {
        @Override
        public State<T> call() {
          double v = fn.apply(state.getPayload(), state.getMappedParams());
          state.setValue(v);
          return state;
        }
      });
    }

    List<Future<State<T>>> r = pool.invokeAll(tasks);

    // zip through the results and find the best one
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

  public void setThreadCount(int threadCount) {
    this.threadCount = threadCount;
    pool = Executors.newFixedThreadPool(threadCount);
  }

  public int getThreadCount() {
    return threadCount;
  }

  public int getPopulationSize() {
    return populationSize;
  }

  public List<State<T>> getPopulation() {
    return population;
  }

  public void close() throws EarlyTerminationException {
    List<Runnable> remainingTasks = pool.shutdownNow();
    if (remainingTasks.size() > 0) {
      throw new EarlyTerminationException("Had to forcefully shut down " + remainingTasks.size() + " tasks");
    }
  }

  public interface Function<T> {
    double apply(T payload, double[] params);
  }

  public static class EarlyTerminationException extends RuntimeException {
    public EarlyTerminationException(String message) {
      super(message);
    }
  }
}
