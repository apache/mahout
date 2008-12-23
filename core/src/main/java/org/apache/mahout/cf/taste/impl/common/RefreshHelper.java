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

package org.apache.mahout.cf.taste.impl.common;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.locks.ReentrantLock;

/**
 * A helper class for implementing {@link Refreshable}.
 */
public final class RefreshHelper implements Refreshable {

  private static final Logger log = LoggerFactory.getLogger(RefreshHelper.class);

  private final List<Refreshable> dependencies;
  private final ReentrantLock refreshLock;
  private final Callable<?> refreshRunnable;

  public RefreshHelper(Callable<?> refreshRunnable) {
    this.dependencies = new ArrayList<Refreshable>(3);
    this.refreshLock = new ReentrantLock();
    this.refreshRunnable = refreshRunnable;
  }

  public void addDependency(Refreshable refreshable) {
    if (refreshable != null) {
      dependencies.add(refreshable);
    }
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    if (!refreshLock.isLocked()) {
      refreshLock.lock();
      try {
        alreadyRefreshed = buildRefreshed(alreadyRefreshed);
        for (Refreshable dependency : dependencies) {
          maybeRefresh(alreadyRefreshed, dependency);
        }
        if (refreshRunnable != null) {
          try {
            refreshRunnable.call();
          } catch (Exception e) {
            log.warn("Unexpected exception while refreshing", e);
          }
        }
      } finally {
        refreshLock.unlock();
      }
    }
  }

  public static Collection<Refreshable> buildRefreshed(Collection<Refreshable> currentAlreadyRefreshed) {
    return currentAlreadyRefreshed == null ? new FastSet<Refreshable>(3) : currentAlreadyRefreshed;
  }

  public static void maybeRefresh(Collection<Refreshable> alreadyRefreshed, Refreshable refreshable) {
    if (!alreadyRefreshed.contains(refreshable)) {
      alreadyRefreshed.add(refreshable);
      refreshable.refresh(alreadyRefreshed);
    }
  }

}
