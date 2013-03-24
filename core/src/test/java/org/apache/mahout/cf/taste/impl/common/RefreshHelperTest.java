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

import com.google.common.collect.Sets;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Test;

import java.util.Collection;

/** Tests {@link RefreshHelper} */
public final class RefreshHelperTest extends TasteTestCase {

  @Test
  public void testCallable() {
    MockRefreshable mock = new MockRefreshable();
    Refreshable helper = new RefreshHelper(mock);
    helper.refresh(null);
    assertEquals(1, mock.getCallCount());
  }

  @Test
  public void testNoCallable() {
    Refreshable helper = new RefreshHelper(null);
    helper.refresh(null);
  }

  @Test
  public void testDependencies() {
    RefreshHelper helper = new RefreshHelper(null);
    MockRefreshable mock1 = new MockRefreshable();
    MockRefreshable mock2 = new MockRefreshable();
    helper.addDependency(mock1);
    helper.addDependency(mock2);
    helper.refresh(null);
    assertEquals(1, mock1.getCallCount());
    assertEquals(1, mock2.getCallCount());
  }

  @Test
  public void testAlreadyRefreshed() {
    RefreshHelper helper = new RefreshHelper(null);
    MockRefreshable mock1 = new MockRefreshable();
    MockRefreshable mock2 = new MockRefreshable();
    helper.addDependency(mock1);
    helper.addDependency(mock2);
    Collection<Refreshable> alreadyRefreshed = Sets.newHashSetWithExpectedSize(1);
    alreadyRefreshed.add(mock1);
    helper.refresh(alreadyRefreshed);
    assertEquals(0, mock1.getCallCount());
    assertEquals(1, mock2.getCallCount());
  }

}
