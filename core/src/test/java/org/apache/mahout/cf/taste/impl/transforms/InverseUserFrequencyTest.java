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

package org.apache.mahout.cf.taste.impl.transforms;

import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;

/** <p>Tests {@link InverseUserFrequency}.</p> */
public final class InverseUserFrequencyTest extends TransformTestCase {

  public void testIUF() throws Exception {
    DataModel dataModel = getDataModel(
            new long[] {1, 2, 3, 4, 5},
            new Double[][] {
                    {0.1},
                    {0.2, 0.3},
                    {0.4, 0.5, 0.6},
                    {0.7, 0.8, 0.9, 1.0},
                    {1.0, 1.0, 1.0, 1.0, 1.0},
            });

    InverseUserFrequency iuf = new InverseUserFrequency(dataModel, 10.0);

    PreferenceArray user5Prefs = dataModel.getPreferencesFromUser(5);

    for (int i = 0; i < 5; i++) {
      Preference pref = user5Prefs.get(i);
      assertNotNull(pref);
      assertEquals(Math.log(5.0 / (double) (5 - i)) / Math.log(iuf.getLogBase()),
          iuf.getTransformedValue(pref),
          EPSILON);
    }

    // Make sure this doesn't throw an exception
    iuf.refresh(null);
  }

}
