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

package org.apache.mahout.classifier.df.data;

import java.util.Random;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.classifier.df.data.Dataset.Attribute;
import org.junit.Test;

public final class DescriptorUtilsTest extends MahoutTestCase {

  /**
   * Test method for
   * {@link org.apache.mahout.classifier.df.data.DescriptorUtils#parseDescriptor(java.lang.CharSequence)}.
   */
  @Test
  public void testParseDescriptor() throws Exception {
    int n = 10;
    int maxnbAttributes = 100;

    Random rng = RandomUtils.getRandom();
    
    for (int nloop = 0; nloop < n; nloop++) {
      int nbAttributes = rng.nextInt(maxnbAttributes) + 1;

      char[] tokens = Utils.randomTokens(rng, nbAttributes);
      Attribute[] attrs = DescriptorUtils.parseDescriptor(Utils.generateDescriptor(tokens));

      // verify that the attributes matches the token list
      assertEquals("attributes size", nbAttributes, attrs.length);

      for (int attr = 0; attr < nbAttributes; attr++) {
        switch (tokens[attr]) {
          case 'I':
            assertTrue(attrs[attr].isIgnored());
            break;
          case 'N':
            assertTrue(attrs[attr].isNumerical());
            break;
          case 'C':
            assertTrue(attrs[attr].isCategorical());
            break;
          case 'L':
            assertTrue(attrs[attr].isLabel());
            break;
        }
      }
    }
  }

  @Test
  public void testGenerateDescription() throws Exception {
    validate("", "");
    validate("I L C C N N N C", "I L C C N N N C");
    validate("I L C C N N N C", "I L 2 C 3 N C");
    validate("I L C C N N N C", " I L  2 C 3 N C ");
    
    try {
      validate("", "I L 2 2 C 2 N C");
      fail("2 consecutive multiplicators");
    } catch (DescriptorException e) {
    }
    
    try {
      validate("", "I L 2 C -2 N C");
      fail("negative multiplicator");
    } catch (DescriptorException e) {
    }
  }
  
  private static void validate(String descriptor, CharSequence description) throws DescriptorException {
    assertEquals(descriptor, DescriptorUtils.generateDescriptor(description));
  }

}
