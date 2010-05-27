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

package org.apache.mahout.utils.nlp.collocations.llr;

import java.io.Serializable;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

/** Group GramKeys based on their Gram, ignoring the secondary sort key, so that all keys with the same Gram are sent
 *  to the same call of the reduce method, sorted in natural order (for GramKeys).
 */
public class GramKeyGroupComparator extends WritableComparator implements Serializable {

  protected GramKeyGroupComparator() {
    super(GramKey.class, true);
  }

  @SuppressWarnings("unchecked")
  @Override
  public int compare(WritableComparable a, WritableComparable b) {
    GramKey gka = (GramKey) a;
    GramKey gkb = (GramKey) b;

    return WritableComparator.compareBytes(gka.getBytes(), 0, gka.getPrimaryLength(),
                                           gkb.getBytes(), 0, gkb.getPrimaryLength());
  }

}
