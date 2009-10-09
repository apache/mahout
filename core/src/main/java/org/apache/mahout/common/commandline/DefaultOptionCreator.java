package org.apache.mahout.common.commandline;

import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;

public class DefaultOptionCreator {
  /**
   * Returns a default command line option for convergence delta specification.
   */
  public static Option convergenceOption(
      final DefaultOptionBuilder obuilder, final ArgumentBuilder abuilder) {
    Option convergenceDeltaOpt = obuilder.withLongName("convergencedelta")
        .withRequired(true).withShortName("v").withArgument(
            abuilder.withName("convergenceDelta").withMinimum(1).withMaximum(1)
                .create()).withDescription("The convergence delta value.")
        .create();
    return convergenceDeltaOpt;
  }

  /**
   * Returns a default command line option for output directory specification.
   */
  public static Option outputOption(final DefaultOptionBuilder obuilder,
      final ArgumentBuilder abuilder) {
    Option outputOpt = obuilder.withLongName("output").withRequired(true)
        .withShortName("o").withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create())
        .withDescription("The directory pathname for output.").create();
    return outputOpt;
  }

  /**
   * Returns a default command line option for input directory specification.
   */
  public static Option inputOption(final DefaultOptionBuilder obuilder,
      final ArgumentBuilder abuilder) {
    Option inputOpt = obuilder
        .withLongName("input")
        .withRequired(true)
        .withShortName("i")
        .withArgument(
            abuilder.withName("input").withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The Path for input Vectors. Must be a SequenceFile of Writable, Vector.")
        .create();
    return inputOpt;
  }

  /**
   * Returns a default command line option for specification of numbers of
   * clusters to create.
   */
  public static Option kOption(DefaultOptionBuilder obuilder,
      ArgumentBuilder abuilder) {
    Option clustersOpt = obuilder
        .withLongName("k")
        .withRequired(false)
        .withArgument(
            abuilder.withName("k").withMinimum(1).withMaximum(1).create())
        .withDescription(
            "The k in k-Means.  If specified, then a random selection of k Vectors will be chosen as the Centroid and written to the clusters output path.")
        .withShortName("k").create();
    return clustersOpt;
  }

  /**
   * Returns a default command line option for specification of max number of
   * iterations.
   */
  public static Option maxIterOption(DefaultOptionBuilder obuilder,
      ArgumentBuilder abuilder) {
    Option maxIterOpt = obuilder
        .withLongName("maxIter")
        .withRequired(true)
        .withShortName("x")
        .withArgument(
            abuilder.withName("maxIter").withMinimum(1).withMaximum(1).create())
        .withDescription("The maximum number of iterations.").create();
    return maxIterOpt;
  }

  /**
   * Returns a default command line option for specification of distance measure
   * class to use.
   */
  public static Option distanceOption(DefaultOptionBuilder obuilder,
      ArgumentBuilder abuilder) {
    Option measureClassOpt = obuilder
        .withLongName("measure")
        .withRequired(true)
        .withShortName("d")
        .withArgument(
            abuilder.withName("measure").withMinimum(1).withMaximum(1).create())
        .withDescription("The classname of the DistanceMeasure.").create();
    return measureClassOpt;
  }

  /**
   * Returns a default command line option for help.
   * */
  public static Option helpOption(DefaultOptionBuilder obuilder) {
    Option helpOpt = obuilder.withLongName("help").
    withDescription("Print out help").withShortName("h").create();
    return helpOpt;
  }

}
