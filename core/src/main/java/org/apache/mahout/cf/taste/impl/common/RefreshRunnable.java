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

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.Refreshable;

import java.util.concurrent.Callable;

/**
 * Simply calls {@linkRefreshable#refresh(java.util.Collection)} on a {@link Refreshable}.
 *
 * @deprecated Not used by RefreshHelper anymore.
 */
public final class RefreshRunnable implements Runnable, Callable<Void> {

  private final Refreshable refreshable;

  public RefreshRunnable(Refreshable refreshable) {
    Preconditions.checkNotNull(refreshable, "Refreshable cannot be null");
    this.refreshable = refreshable;
  }

  @Override
  public void run() {
    refreshable.refresh(null);
  }

  @Override
  public Void call() {
    run();
    return null;
  }

}
