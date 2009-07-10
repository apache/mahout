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

import java.util.Collection;
import java.util.concurrent.Callable;

/** A mock {@link Refreshable} which counts the number of times it has been refreshed, for use in tests. */
final class MockRefreshable implements Refreshable, Callable<Object> {

  private int callCount;

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    call();
  }

  @Override
  public Object call() {
    callCount++;
    return null;
  }

  int getCallCount() {
    return callCount;
  }

}
