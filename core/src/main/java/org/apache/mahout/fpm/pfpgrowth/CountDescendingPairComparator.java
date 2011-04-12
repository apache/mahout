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

package org.apache.mahout.fpm.pfpgrowth;

import java.io.Serializable;
import java.util.Comparator;

import org.apache.mahout.common.Pair;

/**
 * Defines an ordering on {@link Pair}s whose second element is a count. The ordering places those with
 * high count first (that is, descending), and for those of equal count, orders by the first element in the
 * pair, ascending. It is used in several places in the FPM code.
 */
public final class CountDescendingPairComparator<A extends Comparable<? super A>,B extends Comparable<? super B>>
  implements Comparator<Pair<A,B>>, Serializable {

  @Override
  public int compare(Pair<A,B> a, Pair<A,B> b) {
    int ret = b.getSecond().compareTo(a.getSecond());
    if (ret != 0) {
      return ret;
    }
    return a.getFirst().compareTo(b.getFirst());
  }
}
