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

import java.io.IOException;
import java.io.StringWriter;

import com.google.common.base.Charsets;
import com.google.common.io.Resources;

import org.apache.mahout.common.MahoutTestCase;
import org.junit.Test;

/**
 * Test case for {@link Driver}
 */
public class DriverTest extends MahoutTestCase {

  @Test
  public void dictionary() throws IOException {

    ARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterableTest.getVectors("sample-dense.arff", model);
    StringWriter writer = new StringWriter();
    Driver.writeLabelBindings(writer, model, ",");

    String expected = Resources.toString(Resources.getResource("expected-arff-dictionary.csv"), Charsets.UTF_8);

    assertEquals(expected, writer.toString());
  }


  @Test
  public void dictionaryJSON() throws IOException {
    ARFFModel model = new MapBackedARFFModel();
    ARFFVectorIterableTest.getVectors("sample-dense.arff", model);
    StringWriter writer = new StringWriter();
    Driver.writeLabelBindingsJSON(writer, model);
    assertEquals(Resources.toString(Resources.getResource("expected-arff-schema.json"), Charsets.UTF_8), writer.toString());
  }
}
