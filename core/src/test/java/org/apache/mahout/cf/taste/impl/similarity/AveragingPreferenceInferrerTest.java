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

package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericItem;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Item;
import org.apache.mahout.cf.taste.model.User;

import java.util.Collections;

/**
 * <p>Tests {@link AveragingPreferenceInferrer}.</p>
 */
public final class AveragingPreferenceInferrerTest extends TasteTestCase {

  public void testInferrer() throws TasteException {
    User user1 = getUser("test1", 3.0, -2.0, 5.0);
    Item item = new GenericItem<String>("3");
    DataModel model = new GenericDataModel(Collections.singletonList(user1));
    PreferenceInferrer inferrer = new AveragingPreferenceInferrer(model);
    double inferred = inferrer.inferPreference(user1, item);
    assertEquals(2.0, inferred);
  }

}
