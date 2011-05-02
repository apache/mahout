/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.ep;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;

import java.util.Deque;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Implements threaded optimization of an objective function.  The evolving population
 * is updated incrementally as results are received.  One useful feature is that the
 * optimization is inherently time-bounded which is useful for some scheduled operations.
 */
public class ThreadedEvolutionaryProcess {

  private static final PriorityQueue<State<?, ?>> resultPopulation = new PriorityQueue<State<?, ?>>();

  private volatile int taskCount;
  private volatile int processCount;

  private volatile int maxTask;

  private final Deque<State<?, ?>> pending = Lists.newLinkedList();
  private final Set<Future<State<?,?>>> working = Sets.newHashSet();

  private final ExecutorService pool;
  private final ExecutorCompletionService<State<?, ?>> ecs;
  private final int threadCount;
  private final Map<Integer, Mapping> mappingTable = Maps.newHashMap();

  public ThreadedEvolutionaryProcess(int threadCount) {
    this.threadCount = threadCount;
    pool = Executors.newFixedThreadPool(threadCount);
    ecs = new ExecutorCompletionService<State<?, ?>>(pool);
  }

  public void setMap(int i, Mapping m) {
    mappingTable.put(i, m);
  }

  public State<?, ?> optimize(Function f, int dim, long timeLimit, int parentDepth)
    throws InterruptedException, ExecutionException {
    long t0 = System.currentTimeMillis();

    // start with a few points near 0.  These will get transformed
    State<?, ?> s0 = new State(new double[dim], 0.5);
    for (Map.Entry<Integer, Mapping> entry : mappingTable.entrySet()) {
      s0.setMap(entry.getKey(), entry.getValue());
    }

    pending.add(s0);
    while (pending.size() < threadCount) {
      pending.add(s0.mutate());
    }

    // then work until the clock runs out
    do {
      // launch new tasks until we fill the available slots
      while (working.size() < threadCount && !pending.isEmpty()) {
        State<?, ?> next = pending.removeFirst();
        working.add(ecs.submit(new EvalTask(f, next)));
        processCount++;
      }

      // wait for at least one result, then grab any additional results
      Future<State<?, ?>> result = ecs.take();
      while (result != null) {
        State<?, ?> r = result.get();
        resultPopulation.add(r);
        working.remove(result);
        result = ecs.poll();
      }

      // now spawn new pending tasks from the best in recent history
      State<?, ?>[] parents = new State[parentDepth];
      Iterator<State<?, ?>> j = resultPopulation.iterator();
      for (int i = 0; i < parentDepth && j.hasNext(); i++) {
        parents[i] = j.next();
      }

      int k = 0;
      while (pending.size() + working.size() < threadCount) {
        State<?,?> tmp = parents[k++ % parentDepth];
        pending.add(tmp.mutate());
      }
    } while (System.currentTimeMillis() - t0 < timeLimit);

    // wait for last results to dribble in
    while (!working.isEmpty()) {
      Future<State<?, ?>> result = ecs.take();
      working.remove(result);
      resultPopulation.add(result.get());
    }
    pool.shutdown();

    // now do a final pass over the data to get scores
    return resultPopulation.peek();
  }

  
  @Override
  public String toString() {
    return String.format(Locale.ENGLISH,
                         "Launched %d function evaluations\nMaximum threading width was %d", processCount, maxTask);
  }

  public class EvalTask implements Callable<State<?, ?>> {
    private final Function f;
    private final State<?, ?> what;

    public EvalTask(Function f, State<?, ?> what) {
      this.f = f;
      this.what = what;
    }

    /**
     * Computes a result, or throws an exception if unable to do so.
     *
     * @return computed result
     */
    @Override
    public State<?, ?> call() {
      taskCount++;
      maxTask = Math.max(taskCount, maxTask);
      try {
        what.setValue(f.apply(what.getMappedParams()));
        return what;
      } finally {
        taskCount--;
      }
    }
  }

  public interface Function {
    double apply(double[] params);
  }
}
