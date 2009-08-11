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

import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.transforms.PreferenceTransform;

/** <p>Tests {@link ZScore}.</p> */
public final class ZScoreTest extends TransformTestCase {

  public void testOnePref() throws Exception {
    DataModel dataModel = getDataModel(new long[] {1}, new Double[][] {{1.0}});
    PreferenceTransform zScore = new ZScore(dataModel);
    assertEquals(0.0, zScore.getTransformedValue(new GenericPreference(1, 0, 1.0f)), EPSILON);
  }

  public void testAllSame() throws Exception {
    DataModel dataModel = getDataModel(new long[] {1}, new Double[][] {{1.0,1.0,1.0}});
    PreferenceTransform zScore = new ZScore(dataModel);
    assertEquals(0.0, zScore.getTransformedValue(new GenericPreference(1, 0, 1.0f)), EPSILON);
    assertEquals(0.0, zScore.getTransformedValue(new GenericPreference(1, 1, 1.0f)), EPSILON);
    assertEquals(0.0, zScore.getTransformedValue(new GenericPreference(1, 2, 1.0f)), EPSILON);
  }

  public void testStdev() throws Exception {
    DataModel dataModel = getDataModel(new long[] {1}, new Double[][] {{-1.0,-2.0}});
    PreferenceTransform zScore = new ZScore(dataModel);
    assertEquals(0.707107, zScore.getTransformedValue(new GenericPreference(1, 0, -1.0f)), EPSILON);
    assertEquals(-0.707107, zScore.getTransformedValue(new GenericPreference(1, 1, -2.0f)), EPSILON);
  }

  public void testExample() throws Exception {
    DataModel dataModel = getDataModel(new long[] {1}, new Double[][] {{5.0, 7.0, 9.0}});
    PreferenceTransform zScore = new ZScore(dataModel);
    assertEquals(-1.0, zScore.getTransformedValue(new GenericPreference(1, 0, 5.0f)), EPSILON);
    assertEquals(0.0, zScore.getTransformedValue(new GenericPreference(1, 1, 7.0f)), EPSILON);
    assertEquals(1.0, zScore.getTransformedValue(new GenericPreference(1, 2, 9.0f)), EPSILON);
  }

}
