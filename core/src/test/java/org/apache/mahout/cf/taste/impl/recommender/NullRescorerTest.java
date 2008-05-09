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

package org.apache.mahout.cf.taste.impl.recommender;

import junit.framework.TestCase;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.impl.model.GenericUser;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.User;
import org.apache.mahout.cf.taste.recommender.Rescorer;

import java.util.Collections;

/**
 * <p>Tests {@link NullRescorer}.</p>
 */
public final class NullRescorerTest extends TestCase {

  public void testItemRescorer() throws Exception {
    Rescorer<Item> rescorer = NullRescorer.getItemInstance();
    assertNotNull(rescorer);
    Item item = new GenericItem<String>("test");
    assertEquals(1.0, rescorer.rescore(item, 1.0));
    assertEquals(1.0, rescorer.rescore(null, 1.0));
    assertEquals(0.0, rescorer.rescore(item, 0.0));
    assertTrue(Double.isNaN(rescorer.rescore(item, Double.NaN)));
  }

  public void testUserRescorer() throws Exception {
    Rescorer<User> rescorer = NullRescorer.getUserInstance();
    assertNotNull(rescorer);
    User user = new GenericUser<String>("test", Collections.<Preference>emptyList());
    assertEquals(1.0, rescorer.rescore(user, 1.0));
    assertEquals(1.0, rescorer.rescore(null, 1.0));
    assertEquals(0.0, rescorer.rescore(user, 0.0));
    assertTrue(Double.isNaN(rescorer.rescore(user, Double.NaN)));
  }

}
