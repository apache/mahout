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

package org.apache.mahout.cf.taste.impl.model;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Test;

/**
 * Tests {@link GenericDataModel}.
 */
public final class GenericDataModelTest extends TasteTestCase {

  @Test  
  public void testSerialization() throws Exception {
    GenericDataModel model = (GenericDataModel) getDataModel();
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    ObjectOutputStream out = new ObjectOutputStream(baos);
    out.writeObject(model);
    ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
    ObjectInputStream in = new ObjectInputStream(bais);
    GenericDataModel newModel = (GenericDataModel) in.readObject();
    assertEquals(model.getNumItems(), newModel.getNumItems());
    assertEquals(model.getNumUsers(), newModel.getNumUsers());
    assertEquals(model.getPreferencesFromUser(1L), newModel.getPreferencesFromUser(1L));    
    assertEquals(model.getPreferencesForItem(1L), newModel.getPreferencesForItem(1L));
    assertEquals(model.getRawUserData(), newModel.getRawUserData());
  }

  // Lots of other stuff should be tested but is kind of covered by FileDataModelTest

}
