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
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.junit.Test;

/** <p>Tests {@link AveragingPreferenceInferrer}.</p> */
public final class AveragingPreferenceInferrerTest extends TasteTestCase {

  @Test
  public void testInferrer() throws TasteException {
    DataModel model = getDataModel(new long[] {1}, new Double[][] {{3.0,-2.0,5.0}});
    PreferenceInferrer inferrer = new AveragingPreferenceInferrer(model);
    double inferred = inferrer.inferPreference(1, 3);
    assertEquals(2.0, inferred, EPSILON);
  }

}
