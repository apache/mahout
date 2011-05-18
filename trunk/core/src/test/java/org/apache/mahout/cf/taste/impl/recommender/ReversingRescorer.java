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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.Rescorer;

/** <p>Simple {@link Rescorer} which negates the given score, thus reversing order of rankings.</p> */
public final class ReversingRescorer<T> implements Rescorer<T>, IDRescorer {

  @Override
  public double rescore(T thing, double originalScore) {
    return -originalScore;
  }

  @Override
  public boolean isFiltered(T thing) {
    return false;
  }

  @Override
  public double rescore(long ID, double originalScore) {
    return -originalScore;
  }

  @Override
  public boolean isFiltered(long ID) {
    return false;
  }

}
