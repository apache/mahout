/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with this
 * work for additional information regarding copyright ownership. The ASF
 * licenses this file to You under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.apache.mahout.classifier.df.data;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;
@Deprecated
public final class DatasetTest extends MahoutTestCase {

  @Test
  public void jsonEncoding() throws DescriptorException {
    Dataset to = DataLoader.generateDataset("N C I L", true, new String[]{"1 foo 2 3", "4 bar 5 6"});

    // to JSON
    //assertEquals(json, to.toJSON());
    assertEquals(3, to.nbAttributes());
    assertEquals(1, to.getIgnored().length);
    assertEquals(2, to.getIgnored()[0]);
    assertEquals(2, to.getLabelId());
    assertTrue(to.isNumerical(0));

    // from JSON
    Dataset fromJson = Dataset.fromJSON(to.toJSON());
    assertEquals(3, fromJson.nbAttributes());
    assertEquals(1, fromJson.getIgnored().length);
    assertEquals(2, fromJson.getIgnored()[0]);
    assertTrue(fromJson.isNumerical(0));
    
    // read values for a nominal
    assertNotEquals(fromJson.valueOf(1, "bar"), fromJson.valueOf(1, "foo"));
  }

  @Test
  public void jsonEncodingIgnoreFeatures() throws DescriptorException {;
    Dataset to = DataLoader.generateDataset("N C I L", false, new String[]{"1 foo 2 Red", "4 bar 5 Blue"});

    // to JSON
    //assertEquals(json, to.toJSON());
    assertEquals(3, to.nbAttributes());
    assertEquals(1, to.getIgnored().length);
    assertEquals(2, to.getIgnored()[0]);
    assertEquals(2, to.getLabelId());
    assertTrue(to.isNumerical(0));
    assertNotEquals(to.valueOf(1, "bar"), to.valueOf(1, "foo"));
    assertNotEquals(to.valueOf(2, "Red"), to.valueOf(2, "Blue"));

    // from JSON
    Dataset fromJson = Dataset.fromJSON(to.toJSON());
    assertEquals(3, fromJson.nbAttributes());
    assertEquals(1, fromJson.getIgnored().length);
    assertEquals(2, fromJson.getIgnored()[0]);
    assertTrue(fromJson.isNumerical(0));

    // read values for a nominal, one before and one after the ignore feature
    assertNotEquals(fromJson.valueOf(1, "bar"), fromJson.valueOf(1, "foo"));
    assertNotEquals(fromJson.valueOf(2, "Red"), fromJson.valueOf(2, "Blue"));
  }
}
