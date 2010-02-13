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

package org.apache.mahout.common.commandline;

import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;

public final class DefaultOptionCreator {
  
  private DefaultOptionCreator() { }
  
  /**
   * Returns a default command line option for convergence delta specification.
   */
  public static DefaultOptionBuilder convergenceOption() {
    return new DefaultOptionBuilder().withLongName("convergenceDelta").withRequired(true).withShortName("v")
        .withArgument(
          new ArgumentBuilder().withName("convergenceDelta").withMinimum(1).withMaximum(1).create())
        .withDescription("The convergence delta value.");
  }
  
  /**
   * Returns a default command line option for output directory specification.
   */
  public static DefaultOptionBuilder outputOption() {
    return new DefaultOptionBuilder().withLongName("output").withRequired(true).withShortName("o")
        .withArgument(new ArgumentBuilder().withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("The directory pathname for output.");
  }
  
  /**
   * Returns a default command line option for input directory specification.
   */
  public static DefaultOptionBuilder inputOption() {
    return new DefaultOptionBuilder().withLongName("input").withRequired(true).withShortName("i")
        .withArgument(new ArgumentBuilder().withName("input").withMinimum(1).withMaximum(1).create())
        .withDescription("Path to job input directory");
  }
  
  /**
   * Returns a default command line option for specification of numbers of clusters to create.
   */
  public static DefaultOptionBuilder kOption() {
    return new DefaultOptionBuilder()
        .withLongName("k")
        .withRequired(true)
        .withArgument(new ArgumentBuilder().withName("k").withMinimum(1).withMaximum(1).create())
        .withDescription(
          "The k in k-Means. k random Vectors will be chosen as the Centroid and written to the clusters output path.")
        .withShortName("k");
  }
  
  /**
   * Returns a default command line option for specification of max number of iterations.
   */
  public static DefaultOptionBuilder maxIterOption() {
    return new DefaultOptionBuilder().withLongName("maxIter").withRequired(true).withShortName("x")
        .withArgument(new ArgumentBuilder().withName("maxIter").withMinimum(1).withMaximum(1).create())
        .withDescription("The maximum number of iterations.");
  }
  
  /**
   * Returns a default command line option for specification of distance measure class to use.
   */
  public static DefaultOptionBuilder distanceOption() {
    return new DefaultOptionBuilder().withLongName("measure").withRequired(true).withShortName("d")
        .withArgument(new ArgumentBuilder().withName("measure").withMinimum(1).withMaximum(1).create())
        .withDescription("The classname of the DistanceMeasure.");
  }
  
  /**
   * Returns a default command line option for help.
   * */
  public static Option helpOption() {
    return new DefaultOptionBuilder().withLongName("help").withDescription("Print out help").withShortName(
      "h").create();
  }
  
}
