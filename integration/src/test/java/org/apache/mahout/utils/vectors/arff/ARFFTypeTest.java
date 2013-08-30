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

public final class ARFFTypeTest extends MahoutTestCase {

  @Test
  public void removeQuotes() {
    assertNull(ARFFType.removeQuotes(null));
    assertEquals("", ARFFType.removeQuotes("\"\""));
    assertEquals("", ARFFType.removeQuotes("''"));
    assertEquals("", ARFFType.removeQuotes(""));
    assertEquals("", ARFFType.removeQuotes("  "));
    assertEquals("single", ARFFType.removeQuotes("'single'"));
    assertEquals("double", ARFFType.removeQuotes("\"double\""));
    assertEquals("trim", ARFFType.removeQuotes(" trim "));
  }
}
