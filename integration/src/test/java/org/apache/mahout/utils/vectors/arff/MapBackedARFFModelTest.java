/*
 * Copyright 2013 The Apache Software Foundation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.utils.vectors.arff;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

import java.util.Map;


public class MapBackedARFFModelTest extends MahoutTestCase {

  @Test
  public void processNominal() {
    String windy = "windy";
    String breezy = "breezy";

    ARFFModel model = new MapBackedARFFModel();
    model.addNominal(windy, breezy, 77);
    model.addNominal(windy, "strong", 23);
    model.addNominal(windy, "nuking", 55);
    Map<String, Map<String, Integer>> nominalMap = model.getNominalMap();

    assertEquals(1, nominalMap.size());
    Map<String, Integer> windyValues = nominalMap.get(windy);
    assertEquals(77, windyValues.get(breezy).intValue());
  }

  @Test
  public void processBadNumeric() {
    ARFFModel model = new MapBackedARFFModel();
    model.addLabel("b1shkt70694difsmmmdv0ikmoh", 77);
    model.addType(77, ARFFType.REAL);
    assertTrue(Double.isNaN(model.getValue("b1shkt70694difsmmmdv0ikmoh", 77)));
  }

  @Test
  public void processGoodNumeric() {
    ARFFModel model = new MapBackedARFFModel();
    model.addLabel("1234", 77);
    model.addType(77, ARFFType.INTEGER);
    assertTrue(1234 == model.getValue("1234", 77));
    model.addLabel("131.34", 78);
    model.addType(78, ARFFType.REAL);
    assertTrue(131.34 == model.getValue("131.34", 78));
  }
}
