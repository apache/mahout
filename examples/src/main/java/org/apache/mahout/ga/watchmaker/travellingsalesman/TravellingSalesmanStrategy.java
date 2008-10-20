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

package org.apache.mahout.ga.watchmaker.travellingsalesman;

import java.util.Collection;
import java.util.List;

/**
 * Defines methods that must be implemented by classes that provide solutions to
 * the Travelling Salesman problem.
 * 
 * <br>
 * The original code is from <b>the Watchmaker project</b>
 * (https://watchmaker.dev.java.net/).
 */
public interface TravellingSalesmanStrategy {
  /**
   * @return A description of the strategy.
   */
  String getDescription();

  /**
   * Calculates the shortest round trip distance that visits each of the
   * specified cities once and returns to the starting point.
   * 
   * @param cities The destination that must each be visited for the route to be
   *        valid.
   * @param progressListener A call-back for keeping track of the route-finding
   *        algorithm's progress.
   * @return The shortest route found for the given list of destinations.
   */
  List<String> calculateShortestRoute(Collection<String> cities,
      ProgressListener progressListener);
}
