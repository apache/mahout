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
import org.apache.mahout.clustering.dirichlet.models.NormalModelDistribution;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.RandomAccessSparseVector;

public final class DefaultOptionCreator {

  private DefaultOptionCreator() {
  }

  /**
   * Returns a default command line option for help.
   * */
  public static Option helpOption() {
    return new DefaultOptionBuilder().withLongName("help").withDescription("Print out help").withShortName("h").create();
  }

  /**
   * Returns a default command line option for input directory specification.
   */
  public static DefaultOptionBuilder inputOption() {
    return new DefaultOptionBuilder().withLongName("input").withRequired(true).withShortName("i").withArgument(
        new ArgumentBuilder().withName("input").withMinimum(1).withMaximum(1).create()).withDescription(
        "Path to job input directory. Must be a SequenceFile of VectorWritable");
  }

  public static DefaultOptionBuilder clustersInOption() {
    return new DefaultOptionBuilder().withLongName("clusters").withRequired(true).withArgument(
        new ArgumentBuilder().withName("clusters").withMinimum(1).withMaximum(1).create()).withDescription(
        "The path to the initial clusters directory. Must be a SequenceFile of some type of Cluster").withShortName("c");
  }

  /**
   * Returns a default command line option for output directory specification.
   */
  public static DefaultOptionBuilder outputOption() {
    return new DefaultOptionBuilder().withLongName("output").withRequired(true).withShortName("o").withArgument(
        new ArgumentBuilder().withName("output").withMinimum(1).withMaximum(1).create()).withDescription(
        "The directory pathname for output.");
  }

  /**
   * Returns a default command line option for output directory overwriting
   */
  public static DefaultOptionBuilder overwriteOption() {
    return new DefaultOptionBuilder().withLongName("overwrite").withRequired(false).withDescription(
        "If present, overwrite the output directory before running job").withShortName("ow");
  }

  /**
   * Returns a default command line option for clustering specification
   */
  public static DefaultOptionBuilder clusteringOption() {
    return new DefaultOptionBuilder().withLongName("clustering").withRequired(false).withDescription(
        "If present, run clustering after the iterations have taken place").withShortName("cl");
  }

  /**
   * Returns a default command line option for specification of distance measure class to use.
   */
  public static DefaultOptionBuilder distanceMeasureOption() {
    return new DefaultOptionBuilder().withLongName("distanceMeasure").withRequired(false).withShortName("dm").withArgument(
        new ArgumentBuilder().withName("distanceMeasure").withDefault(SquaredEuclideanDistanceMeasure.class.getName()).withMinimum(
            1).withMaximum(1).create()).withDescription("The classname of the DistanceMeasure. Default is SquaredEuclidean");
  }

  public static DefaultOptionBuilder t1Option() {
    return new DefaultOptionBuilder().withLongName("t1").withRequired(true).withArgument(
        new ArgumentBuilder().withName("t1").withMinimum(1).withMaximum(1).create()).withDescription("T1 threshold value")
        .withShortName("t1");
  }

  public static DefaultOptionBuilder t2Option() {
    return new DefaultOptionBuilder().withLongName("t2").withRequired(true).withArgument(
        new ArgumentBuilder().withName("t2").withMinimum(1).withMaximum(1).create()).withDescription("T2 threshold value")
        .withShortName("t2");
  }

  /**
   * Returns a default command line option for specification of max number of iterations.
   */
  public static DefaultOptionBuilder maxIterationsOption() {
    return new DefaultOptionBuilder().withLongName("maxIter").withRequired(true).withShortName("x").withArgument(
        new ArgumentBuilder().withName("maxIter").withMinimum(1).withMaximum(1).create()).withDescription(
        "The maximum number of iterations.");
  }

  /**
   * Returns a default command line option for specification of numbers of clusters to create.
   */
  public static DefaultOptionBuilder kOption() {
    return new DefaultOptionBuilder().withLongName("k").withRequired(true).withArgument(
        new ArgumentBuilder().withName("k").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of clusters to create").withShortName("k");
  }

  /**
   * Returns a default command line option for convergence delta specification.
   */
  public static DefaultOptionBuilder convergenceOption() {
    return new DefaultOptionBuilder().withLongName("convergenceDelta").withRequired(false).withShortName("cd").withArgument(
        new ArgumentBuilder().withName("convergenceDelta").withDefault("0.5").withMinimum(1).withMaximum(1).create())
        .withDescription("The convergence delta value. Default is 0.5");
  }

  /**
   * Returns a default command line option for alpha specification
   */
  public static DefaultOptionBuilder alphaOption() {
    return new DefaultOptionBuilder().withLongName("alpha").withRequired(false).withShortName("m").withArgument(
        new ArgumentBuilder().withName("alpha").withDefault("1.0").withMinimum(1).withMaximum(1).create()).withDescription(
        "The alpha0 value for the DirichletDistribution. Defaults to 1.0");
  }

  /**
   * Returns a default command line option for model distribution class specification
   */
  public static DefaultOptionBuilder modelDistributionOption() {
    return new DefaultOptionBuilder().withLongName("modelDistClass").withRequired(false).withShortName("md").withArgument(
        new ArgumentBuilder().withName("modelDistClass").withDefault(NormalModelDistribution.class.getName()).withMinimum(1)
            .withMaximum(1).create()).withDescription("The ModelDistribution class name. " + "Defaults to NormalModelDistribution");
  }

  /**
   * Returns a default command line option for model prototype class specification
   */
  public static DefaultOptionBuilder modelPrototypeOption() {
    return new DefaultOptionBuilder().withLongName("modelPrototypeClass").withRequired(false).withShortName("mp").withArgument(
        new ArgumentBuilder().withName("prototypeClass").withDefault(RandomAccessSparseVector.class.getName()).withMinimum(1)
            .withMaximum(1).create()).withDescription(
        "The ModelDistribution prototype Vector class name. " + "Defaults to RandomAccessSparseVector");
  }

  /**
   * Returns a default command line option for specifying the number of Mappers
   */
  public static DefaultOptionBuilder numMappersOption() {
    return new DefaultOptionBuilder().withLongName("numMap").withRequired(false).withArgument(
        new ArgumentBuilder().withName("numMap").withDefault("10").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of map tasks. Defaults to 10").withShortName("u");
  }

  /**
   * Returns a default command line option for specifying the max number of reducers
   */
  public static DefaultOptionBuilder numReducersOption() {
    return new DefaultOptionBuilder().withLongName("maxRed").withRequired(false).withShortName("r").withArgument(
        new ArgumentBuilder().withName("maxRed").withDefault("1").withMinimum(1).withMaximum(1).create()).withDescription(
        "The number of reduce tasks. Defaults to 1");
  }

  /**
   * Returns a default command line option for specifying the emitMostLikely 
   */
  public static DefaultOptionBuilder emitMostLikelyOption() {
    return new DefaultOptionBuilder().withLongName("emitMostLikely").withRequired(false).withShortName("e").withArgument(
        new ArgumentBuilder().withName("emitMostLikely").withDefault("false").withMinimum(1).withMaximum(1).create())
        .withDescription("True if clustering should emit the most likely point only, false for threshold clustering");
  }

  /**
   * Returns a default command line option for specifying the clustering threshold value
   */
  public static DefaultOptionBuilder thresholdOption() {
    return new DefaultOptionBuilder().withLongName("threshold").withRequired(false).withShortName("t").withArgument(
        new ArgumentBuilder().withName("threshold").withDefault("0").withMinimum(1).withMaximum(1).create()).withDescription(
        "The pdf threshold used for cluster determination. Default is 0");
  }

  /**
   * Returns a default command line option for specifying the FuzzyKMeans coefficient normalization factor, 'm'
   */
  public static DefaultOptionBuilder mOption() {
    return new DefaultOptionBuilder().withLongName("m").withRequired(true).withArgument(
        new ArgumentBuilder().withName("m").withMinimum(1).withMaximum(1).create()).withDescription(
        "coefficient normalization factor, must be greater than 1").withShortName("m");
  }

  /**
   * Returns a default command line option for specifying that the MeanShift input directory already contains Canopies vs. Vectors
   */
  public static DefaultOptionBuilder inputIsCanopiesOption() {
    return new DefaultOptionBuilder().withLongName("inputIsCanopies").withRequired(false).withShortName("ic").withArgument(
        new ArgumentBuilder().withName("inputIsCanopies").withMinimum(1).withMaximum(1).create()).withDescription(
        "If present, the input directory already contains MeanShiftCanopies");
  }

}
