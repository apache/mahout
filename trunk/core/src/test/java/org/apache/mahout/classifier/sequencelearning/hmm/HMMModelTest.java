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

package org.apache.mahout.classifier.sequencelearning.hmm;

import org.junit.Test;

public class HMMModelTest extends HMMTestBase {

  @Test
  public void testRandomModelGeneration() {
    // make sure we generate a valid random model
    HmmModel model = new HmmModel(10, 20);
    // check whether the model is valid
    HmmUtils.validate(model);
  }

  @Test
  public void testSerialization() {
    String serialized = getModel().toJson();
    HmmModel model2 = HmmModel.fromJson(serialized);
    String serialized2 = model2.toJson();
    // since there are no equals methods for the underlying objects, we
    // check identity via the serialization string
    assertEquals(serialized, serialized2);
  }

}
