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

package org.apache.mahout.cf.taste.eval;

/**
 * <p>Implementations encapsulate information retrieval-related statistics about a {@link
 * org.apache.mahout.cf.taste.recommender.Recommender}'s recommendations.</p>
 *
 * <p>See <a href="http://en.wikipedia.org/wiki/Information_retrieval">Information retrieval</a>.</p>
 */
public interface IRStatistics {

  /** <p>See <a href="http://en.wikipedia.org/wiki/Information_retrieval#Precision">Precision</a>.</p> */
  double getPrecision();

  /** <p>See <a href="http://en.wikipedia.org/wiki/Information_retrieval#Recall">Recall</a>.</p> */
  double getRecall();

  /** <p>See <a href="http://en.wikipedia.org/wiki/Information_retrieval#Fall-Out">Fall-Out</a>.</p> */
  double getFallOut();

  /** <p>See <a href="http://en.wikipedia.org/wiki/Information_retrieval#F-measure">F-measure</a>.</p> */
  double getF1Measure();

  /** <p>See <a href="http://en.wikipedia.org/wiki/Information_retrieval#F-measure">F-measure</a>.</p> */
  double getFNMeasure(double n);

}
