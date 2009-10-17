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

public class DefaultOptionCreator {
  private DefaultOptionCreator() {
  }

  /**
   * Returns a default command line option for convergence delta specification.
   */
  public static DefaultOptionBuilder convergenceOption(
      final DefaultOptionBuilder obuilder, final ArgumentBuilder abuilder) {
    return obuilder.withLongName("convergencedelta")
        .withRequired(true).withShortName("v").withArgument(
            abuilder.withName("convergenceDelta").withMinimum(1).withMaximum(1)
                .create()).withDescription("The convergence delta value.");
  }

  /**
   * Returns a default command line option for output directory specification.
   */
  public static DefaultOptionBuilder outputOption(final DefaultOptionBuilder obuilder,
      final ArgumentBuilder abuilder) {
    return obuilder.withLongName("output").withRequired(true)
        .withShortName("o").withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("The directory pathname for output.");
  }

  /**
   * Returns a default command line option for input directory specification.
   */
  public static DefaultOptionBuilder inputOption(final DefaultOptionBuilder obuilder,
      final ArgumentBuilder abuilder) {
    return obuilder
        .withLongName("input")
        .withRequired(true)
        .withShortName("i")
        .withArgument(
            abuilder.withName("input").withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The Path for input Vectors. Must be a SequenceFile of Writable, Vector.");
  }

  /**
   * Returns a default command line option for specification of numbers of
   * clusters to create.
   */
  public static DefaultOptionBuilder kOption(DefaultOptionBuilder obuilder,
      ArgumentBuilder abuilder) {
    return obuilder
        .withLongName("k")
        .withRequired(true)
        .withArgument(
            abuilder.withName("k").withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The k in k-Means.  If specified, then a random selection of k Vectors will be chosen as the Centroid and written to the clusters output path.")
        .withShortName("k");
  }

  /**
   * Returns a default command line option for specification of max number of
   * iterations.
   */
  public static DefaultOptionBuilder maxIterOption(DefaultOptionBuilder obuilder,
      ArgumentBuilder abuilder) {
    return obuilder
        .withLongName("maxIter")
        .withRequired(true)
        .withShortName("x")
        .withArgument(
            abuilder.withName("maxIter").withMinimum(1).withMaximum(1).create())
        .withDescription("The maximum number of iterations.");
  }

  /**
   * Returns a default command line option for specification of distance measure
   * class to use.
   */
  public static DefaultOptionBuilder distanceOption(DefaultOptionBuilder obuilder,
      ArgumentBuilder abuilder) {
    return obuilder
        .withLongName("measure")
        .withRequired(true)
        .withShortName("d")
        .withArgument(
            abuilder.withName("measure").withMinimum(1).withMaximum(1).create())
        .withDescription("The classname of the DistanceMeasure.");
  }

  /**
   * Returns a default command line option for help.
   * */
  public static Option helpOption(DefaultOptionBuilder obuilder) {
    return obuilder.withLongName("help").withDescription("Print out help").withShortName("h").create();
  }

}
