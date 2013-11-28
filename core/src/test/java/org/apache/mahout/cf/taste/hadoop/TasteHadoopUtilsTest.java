/*
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

package org.apache.mahout.cf.taste.hadoop;

import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.junit.Test;

/** <p>Tests {@link TasteHadoopUtils}.</p> */
public class TasteHadoopUtilsTest extends TasteTestCase {
	
  @Test
  public void testWithinRange() {
    assertTrue(TasteHadoopUtils.idToIndex(0) >= 0);
    assertTrue(TasteHadoopUtils.idToIndex(0) < Integer.MAX_VALUE);

    assertTrue(TasteHadoopUtils.idToIndex(1) >= 0);
    assertTrue(TasteHadoopUtils.idToIndex(1) < Integer.MAX_VALUE);
		
    assertTrue(TasteHadoopUtils.idToIndex(Long.MAX_VALUE) >= 0);
    assertTrue(TasteHadoopUtils.idToIndex(Long.MAX_VALUE) < Integer.MAX_VALUE);
		
    assertTrue(TasteHadoopUtils.idToIndex(Integer.MAX_VALUE) >= 0);
    assertTrue(TasteHadoopUtils.idToIndex(Integer.MAX_VALUE) < Integer.MAX_VALUE);
  }
}
