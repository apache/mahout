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

package org.apache.mahout.cf.taste.impl.correlation;

import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.User;

/**
 * <p>Tests {@link TanimotoCoefficientCorrelation}.</p>
 */
public final class TanimotoCoefficientCorrelationTest extends CorrelationTestCase {

  public void testNoCorrelation1() throws Exception {
    User user1 = getUser("test1");
    User user2 = getUser("test2");
    DataModel dataModel = getDataModel(user1, user2);
    double correlation = new TanimotoCoefficientCorrelation(dataModel).userCorrelation(user1, user2);
    assertTrue(Double.isNaN(correlation));
  }

  public void testNoCorrelation2() throws Exception {
    User user1 = getUser("test1");
    User user2 = getUser("test2", 1.0);
    DataModel dataModel = getDataModel(user1, user2);
    double correlation = new TanimotoCoefficientCorrelation(dataModel).userCorrelation(user1, user2);
    assertCorrelationEquals(0.0, correlation);
  }

  public void testNoCorrelation() throws Exception {
    User user1 = getUser("test1", null, 2.0, 3.0);
    User user2 = getUser("test2", 1.0);
    DataModel dataModel = getDataModel(user1, user2);
    double correlation = new TanimotoCoefficientCorrelation(dataModel).userCorrelation(user1, user2);
    assertCorrelationEquals(0.0, correlation);
  }

  public void testFullCorrelation1() throws Exception {
    User user1 = getUser("test1", 1.0);
    User user2 = getUser("test2", 1.0);
    DataModel dataModel = getDataModel(user1, user2);
    double correlation = new TanimotoCoefficientCorrelation(dataModel).userCorrelation(user1, user2);
    assertCorrelationEquals(1.0, correlation);
  }

  public void testFullCorrelation2() throws Exception {
    User user1 = getUser("test1", 1.0, 2.0, 3.0);
    User user2 = getUser("test2", 1.0);
    DataModel dataModel = getDataModel(user1, user2);
    double correlation = new TanimotoCoefficientCorrelation(dataModel).userCorrelation(user1, user2);
    assertCorrelationEquals(0.3333333333333333, correlation);
  }

  public void testCorrelation1() throws Exception {
    User user1 = getUser("test1", null, 2.0, 3.0);
    User user2 = getUser("test2", 1.0, 1.0);
    DataModel dataModel = getDataModel(user1, user2);
    double correlation = new TanimotoCoefficientCorrelation(dataModel).userCorrelation(user1, user2);
    assertEquals(0.3333333333333333, correlation);
  }

  public void testCorrelation2() throws Exception {
    User user1 = getUser("test1", null, 2.0, 3.0, 1.0);
    User user2 = getUser("test2", 1.0, 1.0, null, 0.0);
    DataModel dataModel = getDataModel(user1, user2);
    double correlation = new TanimotoCoefficientCorrelation(dataModel).userCorrelation(user1, user2);
    assertEquals(0.5, correlation);
  }

  public void testRefresh() {
    // Make sure this doesn't throw an exception
    new TanimotoCoefficientCorrelation(getDataModel()).refresh();
  }

}