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

package org.apache.mahout.math.hadoop.similarity.cooccurrence;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.mapreduce.VectorSumCombiner;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasure;
import org.apache.mahout.math.map.OpenIntIntHashMap;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

public class RowSimilarityJob extends AbstractJob {

  public static final double NO_THRESHOLD = Double.MIN_VALUE;
  public static final long NO_FIXED_RANDOM_SEED = Long.MIN_VALUE;

  private static final String SIMILARITY_CLASSNAME = RowSimilarityJob.class + ".distributedSimilarityClassname";
  private static final String NUMBER_OF_COLUMNS = RowSimilarityJob.class + ".numberOfColumns";
  private static final String MAX_SIMILARITIES_PER_ROW = RowSimilarityJob.class + ".maxSimilaritiesPerRow";
  private static final String EXCLUDE_SELF_SIMILARITY = RowSimilarityJob.class + ".excludeSelfSimilarity";

  private static final String THRESHOLD = RowSimilarityJob.class + ".threshold";
  private static final String NORMS_PATH = RowSimilarityJob.class + ".normsPath";
  private static final String MAXVALUES_PATH = RowSimilarityJob.class + ".maxWeightsPath";

  private static final String NUM_NON_ZERO_ENTRIES_PATH = RowSimilarityJob.class + ".nonZeroEntriesPath";
  private static final int DEFAULT_MAX_SIMILARITIES_PER_ROW = 100;

  private static final String OBSERVATIONS_PER_COLUMN_PATH = RowSimilarityJob.class + ".observationsPerColumnPath";

  private static final String MAX_OBSERVATIONS_PER_ROW = RowSimilarityJob.class + ".maxObservationsPerRow";
  private static final String MAX_OBSERVATIONS_PER_COLUMN = RowSimilarityJob.class + ".maxObservationsPerColumn";
  private static final String RANDOM_SEED = RowSimilarityJob.class + ".randomSeed";

  private static final int DEFAULT_MAX_OBSERVATIONS_PER_ROW = 500;
  private static final int DEFAULT_MAX_OBSERVATIONS_PER_COLUMN = 500;

  private static final int NORM_VECTOR_MARKER = Integer.MIN_VALUE;
  private static final int MAXVALUE_VECTOR_MARKER = Integer.MIN_VALUE + 1;
  private static final int NUM_NON_ZERO_ENTRIES_VECTOR_MARKER = Integer.MIN_VALUE + 2;

  enum Counters { ROWS, USED_OBSERVATIONS, NEGLECTED_OBSERVATIONS, COOCCURRENCES, PRUNED_COOCCURRENCES }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RowSimilarityJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("numberOfColumns", "r", "Number of columns in the input matrix", false);
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate, alternatively use "
        + "one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')');
    addOption("maxSimilaritiesPerRow", "m", "Number of maximum similarities per row (default: "
        + DEFAULT_MAX_SIMILARITIES_PER_ROW + ')', String.valueOf(DEFAULT_MAX_SIMILARITIES_PER_ROW));
    addOption("excludeSelfSimilarity", "ess", "compute similarity of rows to themselves?", String.valueOf(false));
    addOption("threshold", "tr", "discard row pairs with a similarity value below this", false);
    addOption("maxObservationsPerRow", null, "sample rows down to this number of entries",
        String.valueOf(DEFAULT_MAX_OBSERVATIONS_PER_ROW));
    addOption("maxObservationsPerColumn", null, "sample columns down to this number of entries",
        String.valueOf(DEFAULT_MAX_OBSERVATIONS_PER_COLUMN));
    addOption("randomSeed", null, "use this seed for sampling", false);
    addOption(DefaultOptionCreator.overwriteOption().create());

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    int numberOfColumns;

    if (hasOption("numberOfColumns")) {
      // Number of columns explicitly specified via CLI
      numberOfColumns = Integer.parseInt(getOption("numberOfColumns"));
    } else {
      // else get the number of columns by determining the cardinality of a vector in the input matrix
      numberOfColumns = getDimensions(getInputPath());
    }

    String similarityClassnameArg = getOption("similarityClassname");
    String similarityClassname;
    try {
      similarityClassname = VectorSimilarityMeasures.valueOf(similarityClassnameArg).getClassname();
    } catch (IllegalArgumentException iae) {
      similarityClassname = similarityClassnameArg;
    }

    // Clear the output and temp paths if the overwrite option has been set
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      // Clear the temp path
      HadoopUtil.delete(getConf(), getTempPath());
      // Clear the output path
      HadoopUtil.delete(getConf(), getOutputPath());
    }

    int maxSimilaritiesPerRow = Integer.parseInt(getOption("maxSimilaritiesPerRow"));
    boolean excludeSelfSimilarity = Boolean.parseBoolean(getOption("excludeSelfSimilarity"));
    double threshold = hasOption("threshold")
        ? Double.parseDouble(getOption("threshold")) : NO_THRESHOLD;
    long randomSeed = hasOption("randomSeed")
        ? Long.parseLong(getOption("randomSeed")) : NO_FIXED_RANDOM_SEED;

    int maxObservationsPerRow = Integer.parseInt(getOption("maxObservationsPerRow"));
    int maxObservationsPerColumn = Integer.parseInt(getOption("maxObservationsPerColumn"));

    Path weightsPath = getTempPath("weights");
    Path normsPath = getTempPath("norms.bin");
    Path numNonZeroEntriesPath = getTempPath("numNonZeroEntries.bin");
    Path maxValuesPath = getTempPath("maxValues.bin");
    Path pairwiseSimilarityPath = getTempPath("pairwiseSimilarity");

    Path observationsPerColumnPath = getTempPath("observationsPerColumn.bin");

    AtomicInteger currentPhase = new AtomicInteger();

    Job countObservations = prepareJob(getInputPath(), getTempPath("notUsed"), CountObservationsMapper.class,
        NullWritable.class, VectorWritable.class, SumObservationsReducer.class, NullWritable.class,
        VectorWritable.class);
    countObservations.setCombinerClass(VectorSumCombiner.class);
    countObservations.getConfiguration().set(OBSERVATIONS_PER_COLUMN_PATH, observationsPerColumnPath.toString());
    countObservations.setNumReduceTasks(1);
    countObservations.waitForCompletion(true);

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job normsAndTranspose = prepareJob(getInputPath(), weightsPath, VectorNormMapper.class, IntWritable.class,
          VectorWritable.class, MergeVectorsReducer.class, IntWritable.class, VectorWritable.class);
      normsAndTranspose.setCombinerClass(MergeVectorsCombiner.class);
      Configuration normsAndTransposeConf = normsAndTranspose.getConfiguration();
      normsAndTransposeConf.set(THRESHOLD, String.valueOf(threshold));
      normsAndTransposeConf.set(NORMS_PATH, normsPath.toString());
      normsAndTransposeConf.set(NUM_NON_ZERO_ENTRIES_PATH, numNonZeroEntriesPath.toString());
      normsAndTransposeConf.set(MAXVALUES_PATH, maxValuesPath.toString());
      normsAndTransposeConf.set(SIMILARITY_CLASSNAME, similarityClassname);
      normsAndTransposeConf.set(OBSERVATIONS_PER_COLUMN_PATH, observationsPerColumnPath.toString());
      normsAndTransposeConf.set(MAX_OBSERVATIONS_PER_ROW, String.valueOf(maxObservationsPerRow));
      normsAndTransposeConf.set(MAX_OBSERVATIONS_PER_COLUMN, String.valueOf(maxObservationsPerColumn));
      normsAndTransposeConf.set(RANDOM_SEED, String.valueOf(randomSeed));

      boolean succeeded = normsAndTranspose.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job pairwiseSimilarity = prepareJob(weightsPath, pairwiseSimilarityPath, CooccurrencesMapper.class,
          IntWritable.class, VectorWritable.class, SimilarityReducer.class, IntWritable.class, VectorWritable.class);
      pairwiseSimilarity.setCombinerClass(VectorSumReducer.class);
      Configuration pairwiseConf = pairwiseSimilarity.getConfiguration();
      pairwiseConf.set(THRESHOLD, String.valueOf(threshold));
      pairwiseConf.set(NORMS_PATH, normsPath.toString());
      pairwiseConf.set(NUM_NON_ZERO_ENTRIES_PATH, numNonZeroEntriesPath.toString());
      pairwiseConf.set(MAXVALUES_PATH, maxValuesPath.toString());
      pairwiseConf.set(SIMILARITY_CLASSNAME, similarityClassname);
      pairwiseConf.setInt(NUMBER_OF_COLUMNS, numberOfColumns);
      pairwiseConf.setBoolean(EXCLUDE_SELF_SIMILARITY, excludeSelfSimilarity);
      boolean succeeded = pairwiseSimilarity.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job asMatrix = prepareJob(pairwiseSimilarityPath, getOutputPath(), UnsymmetrifyMapper.class,
          IntWritable.class, VectorWritable.class, MergeToTopKSimilaritiesReducer.class, IntWritable.class,
          VectorWritable.class);
      asMatrix.setCombinerClass(MergeToTopKSimilaritiesReducer.class);
      asMatrix.getConfiguration().setInt(MAX_SIMILARITIES_PER_ROW, maxSimilaritiesPerRow);
      boolean succeeded = asMatrix.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
    }

    return 0;
  }

  public static class CountObservationsMapper extends Mapper<IntWritable,VectorWritable,NullWritable,VectorWritable> {

    private Vector columnCounts = new RandomAccessSparseVector(Integer.MAX_VALUE);

    @Override
    protected void map(IntWritable rowIndex, VectorWritable rowVectorWritable, Context ctx)
      throws IOException, InterruptedException {

      Vector row = rowVectorWritable.get();
      for (Vector.Element elem : row.nonZeroes()) {
        columnCounts.setQuick(elem.index(), columnCounts.getQuick(elem.index()) + 1);
      }
    }

    @Override
    protected void cleanup(Context ctx) throws IOException, InterruptedException {
      ctx.write(NullWritable.get(), new VectorWritable(columnCounts));
    }
  }

  public static class SumObservationsReducer extends Reducer<NullWritable,VectorWritable,NullWritable,VectorWritable> {
    @Override
    protected void reduce(NullWritable nullWritable, Iterable<VectorWritable> partialVectors, Context ctx)
    throws IOException, InterruptedException {
      Vector counts = Vectors.sum(partialVectors.iterator());
      Vectors.write(counts, new Path(ctx.getConfiguration().get(OBSERVATIONS_PER_COLUMN_PATH)), ctx.getConfiguration());
    }
  }

  public static class VectorNormMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private VectorSimilarityMeasure similarity;
    private Vector norms;
    private Vector nonZeroEntries;
    private Vector maxValues;
    private double threshold;

    private OpenIntIntHashMap observationsPerColumn;
    private int maxObservationsPerRow;
    private int maxObservationsPerColumn;

    private Random random;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {

      Configuration conf = ctx.getConfiguration();

      similarity = ClassUtils.instantiateAs(conf.get(SIMILARITY_CLASSNAME), VectorSimilarityMeasure.class);
      norms = new RandomAccessSparseVector(Integer.MAX_VALUE);
      nonZeroEntries = new RandomAccessSparseVector(Integer.MAX_VALUE);
      maxValues = new RandomAccessSparseVector(Integer.MAX_VALUE);
      threshold = Double.parseDouble(conf.get(THRESHOLD));

      observationsPerColumn = Vectors.readAsIntMap(new Path(conf.get(OBSERVATIONS_PER_COLUMN_PATH)), conf);
      maxObservationsPerRow = conf.getInt(MAX_OBSERVATIONS_PER_ROW, DEFAULT_MAX_OBSERVATIONS_PER_ROW);
      maxObservationsPerColumn = conf.getInt(MAX_OBSERVATIONS_PER_COLUMN, DEFAULT_MAX_OBSERVATIONS_PER_COLUMN);

      long seed = Long.parseLong(conf.get(RANDOM_SEED));
      if (seed == NO_FIXED_RANDOM_SEED) {
        random = RandomUtils.getRandom();
      } else {
        random = RandomUtils.getRandom(seed);
      }
    }

    private Vector sampleDown(Vector rowVector, Context ctx) {

      int observationsPerRow = rowVector.getNumNondefaultElements();
      double rowSampleRate = (double) Math.min(maxObservationsPerRow, observationsPerRow) / (double) observationsPerRow;

      Vector downsampledRow = rowVector.like();
      long usedObservations = 0;
      long neglectedObservations = 0;

      for (Vector.Element elem : rowVector.nonZeroes()) {

        int columnCount = observationsPerColumn.get(elem.index());
        double columnSampleRate = (double) Math.min(maxObservationsPerColumn, columnCount) / (double) columnCount;

        if (random.nextDouble() <= Math.min(rowSampleRate, columnSampleRate)) {
          downsampledRow.setQuick(elem.index(), elem.get());
          usedObservations++;
        } else {
          neglectedObservations++;
        }

      }

      ctx.getCounter(Counters.USED_OBSERVATIONS).increment(usedObservations);
      ctx.getCounter(Counters.NEGLECTED_OBSERVATIONS).increment(neglectedObservations);

      return downsampledRow;
    }

    @Override
    protected void map(IntWritable row, VectorWritable vectorWritable, Context ctx)
      throws IOException, InterruptedException {

      Vector sampledRowVector = sampleDown(vectorWritable.get(), ctx);

      Vector rowVector = similarity.normalize(sampledRowVector);

      int numNonZeroEntries = 0;
      double maxValue = Double.MIN_VALUE;

      for (Vector.Element element : rowVector.nonZeroes()) {
        RandomAccessSparseVector partialColumnVector = new RandomAccessSparseVector(Integer.MAX_VALUE);
        partialColumnVector.setQuick(row.get(), element.get());
        ctx.write(new IntWritable(element.index()), new VectorWritable(partialColumnVector));

        numNonZeroEntries++;
        if (maxValue < element.get()) {
          maxValue = element.get();
        }
      }

      if (threshold != NO_THRESHOLD) {
        nonZeroEntries.setQuick(row.get(), numNonZeroEntries);
        maxValues.setQuick(row.get(), maxValue);
      }
      norms.setQuick(row.get(), similarity.norm(rowVector));

      ctx.getCounter(Counters.ROWS).increment(1);
    }

    @Override
    protected void cleanup(Context ctx) throws IOException, InterruptedException {
      ctx.write(new IntWritable(NORM_VECTOR_MARKER), new VectorWritable(norms));
      ctx.write(new IntWritable(NUM_NON_ZERO_ENTRIES_VECTOR_MARKER), new VectorWritable(nonZeroEntries));
      ctx.write(new IntWritable(MAXVALUE_VECTOR_MARKER), new VectorWritable(maxValues));
    }
  }

  private static class MergeVectorsCombiner extends Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {
    @Override
    protected void reduce(IntWritable row, Iterable<VectorWritable> partialVectors, Context ctx)
      throws IOException, InterruptedException {
      ctx.write(row, new VectorWritable(Vectors.merge(partialVectors)));
    }
  }

  public static class MergeVectorsReducer extends Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private Path normsPath;
    private Path numNonZeroEntriesPath;
    private Path maxValuesPath;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      normsPath = new Path(ctx.getConfiguration().get(NORMS_PATH));
      numNonZeroEntriesPath = new Path(ctx.getConfiguration().get(NUM_NON_ZERO_ENTRIES_PATH));
      maxValuesPath = new Path(ctx.getConfiguration().get(MAXVALUES_PATH));
    }

    @Override
    protected void reduce(IntWritable row, Iterable<VectorWritable> partialVectors, Context ctx)
      throws IOException, InterruptedException {
      Vector partialVector = Vectors.merge(partialVectors);

      if (row.get() == NORM_VECTOR_MARKER) {
        Vectors.write(partialVector, normsPath, ctx.getConfiguration());
      } else if (row.get() == MAXVALUE_VECTOR_MARKER) {
        Vectors.write(partialVector, maxValuesPath, ctx.getConfiguration());
      } else if (row.get() == NUM_NON_ZERO_ENTRIES_VECTOR_MARKER) {
        Vectors.write(partialVector, numNonZeroEntriesPath, ctx.getConfiguration(), true);
      } else {
        ctx.write(row, new VectorWritable(partialVector));
      }
    }
  }


  public static class CooccurrencesMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private VectorSimilarityMeasure similarity;

    private OpenIntIntHashMap numNonZeroEntries;
    private Vector maxValues;
    private double threshold;

    private static final Comparator<Vector.Element> BY_INDEX = new Comparator<Vector.Element>() {
      @Override
      public int compare(Vector.Element one, Vector.Element two) {
        return Ints.compare(one.index(), two.index());
      }
    };

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      similarity = ClassUtils.instantiateAs(ctx.getConfiguration().get(SIMILARITY_CLASSNAME),
          VectorSimilarityMeasure.class);
      numNonZeroEntries = Vectors.readAsIntMap(new Path(ctx.getConfiguration().get(NUM_NON_ZERO_ENTRIES_PATH)),
          ctx.getConfiguration());
      maxValues = Vectors.read(new Path(ctx.getConfiguration().get(MAXVALUES_PATH)), ctx.getConfiguration());
      threshold = Double.parseDouble(ctx.getConfiguration().get(THRESHOLD));
    }

    private boolean consider(Vector.Element occurrenceA, Vector.Element occurrenceB) {
      int numNonZeroEntriesA = numNonZeroEntries.get(occurrenceA.index());
      int numNonZeroEntriesB = numNonZeroEntries.get(occurrenceB.index());

      double maxValueA = maxValues.get(occurrenceA.index());
      double maxValueB = maxValues.get(occurrenceB.index());

      return similarity.consider(numNonZeroEntriesA, numNonZeroEntriesB, maxValueA, maxValueB, threshold);
    }

    @Override
    protected void map(IntWritable column, VectorWritable occurrenceVector, Context ctx)
      throws IOException, InterruptedException {
      Vector.Element[] occurrences = Vectors.toArray(occurrenceVector);
      Arrays.sort(occurrences, BY_INDEX);

      int cooccurrences = 0;
      int prunedCooccurrences = 0;
      for (int n = 0; n < occurrences.length; n++) {
        Vector.Element occurrenceA = occurrences[n];
        Vector dots = new RandomAccessSparseVector(Integer.MAX_VALUE);
        for (int m = n; m < occurrences.length; m++) {
          Vector.Element occurrenceB = occurrences[m];
          if (threshold == NO_THRESHOLD || consider(occurrenceA, occurrenceB)) {
            dots.setQuick(occurrenceB.index(), similarity.aggregate(occurrenceA.get(), occurrenceB.get()));
            cooccurrences++;
          } else {
            prunedCooccurrences++;
          }
        }
        ctx.write(new IntWritable(occurrenceA.index()), new VectorWritable(dots));
      }
      ctx.getCounter(Counters.COOCCURRENCES).increment(cooccurrences);
      ctx.getCounter(Counters.PRUNED_COOCCURRENCES).increment(prunedCooccurrences);
    }
  }


  public static class SimilarityReducer extends Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private VectorSimilarityMeasure similarity;
    private int numberOfColumns;
    private boolean excludeSelfSimilarity;
    private Vector norms;
    private double treshold;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      similarity = ClassUtils.instantiateAs(ctx.getConfiguration().get(SIMILARITY_CLASSNAME),
          VectorSimilarityMeasure.class);
      numberOfColumns = ctx.getConfiguration().getInt(NUMBER_OF_COLUMNS, -1);
      Preconditions.checkArgument(numberOfColumns > 0, "Number of columns must be greater then 0! But numberOfColumns = " + numberOfColumns);
      excludeSelfSimilarity = ctx.getConfiguration().getBoolean(EXCLUDE_SELF_SIMILARITY, false);
      norms = Vectors.read(new Path(ctx.getConfiguration().get(NORMS_PATH)), ctx.getConfiguration());
      treshold = Double.parseDouble(ctx.getConfiguration().get(THRESHOLD));
    }

    @Override
    protected void reduce(IntWritable row, Iterable<VectorWritable> partialDots, Context ctx)
      throws IOException, InterruptedException {
      Iterator<VectorWritable> partialDotsIterator = partialDots.iterator();
      Vector dots = partialDotsIterator.next().get();
      while (partialDotsIterator.hasNext()) {
        Vector toAdd = partialDotsIterator.next().get();
        for (Element nonZeroElement : toAdd.nonZeroes()) {
          dots.setQuick(nonZeroElement.index(), dots.getQuick(nonZeroElement.index()) + nonZeroElement.get());
        }
      }

      Vector similarities = dots.like();
      double normA = norms.getQuick(row.get());
      for (Element b : dots.nonZeroes()) {
        double similarityValue = similarity.similarity(b.get(), normA, norms.getQuick(b.index()), numberOfColumns);
        if (similarityValue >= treshold) {
          similarities.set(b.index(), similarityValue);
        }
      }
      if (excludeSelfSimilarity) {
        similarities.setQuick(row.get(), 0);
      }
      ctx.write(row, new VectorWritable(similarities));
    }
  }

  public static class UnsymmetrifyMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable>  {

    private int maxSimilaritiesPerRow;

    @Override
    protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
      maxSimilaritiesPerRow = ctx.getConfiguration().getInt(MAX_SIMILARITIES_PER_ROW, 0);
      Preconditions.checkArgument(maxSimilaritiesPerRow > 0, "Maximum number of similarities per row must be greater then 0!");
    }

    @Override
    protected void map(IntWritable row, VectorWritable similaritiesWritable, Context ctx)
      throws IOException, InterruptedException {
      Vector similarities = similaritiesWritable.get();
      // For performance, the creation of transposedPartial is moved out of the while loop and it is reused inside
      Vector transposedPartial = new RandomAccessSparseVector(similarities.size(), 1);
      TopElementsQueue topKQueue = new TopElementsQueue(maxSimilaritiesPerRow);
      for (Element nonZeroElement : similarities.nonZeroes()) {
        MutableElement top = topKQueue.top();
        double candidateValue = nonZeroElement.get();
        if (candidateValue > top.get()) {
          top.setIndex(nonZeroElement.index());
          top.set(candidateValue);
          topKQueue.updateTop();
        }

        transposedPartial.setQuick(row.get(), candidateValue);
        ctx.write(new IntWritable(nonZeroElement.index()), new VectorWritable(transposedPartial));
        transposedPartial.setQuick(row.get(), 0.0);
      }
      Vector topKSimilarities = new RandomAccessSparseVector(similarities.size(), maxSimilaritiesPerRow);
      for (Vector.Element topKSimilarity : topKQueue.getTopElements()) {
        topKSimilarities.setQuick(topKSimilarity.index(), topKSimilarity.get());
      }
      ctx.write(row, new VectorWritable(topKSimilarities));
    }
  }

  public static class MergeToTopKSimilaritiesReducer
      extends Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private int maxSimilaritiesPerRow;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      maxSimilaritiesPerRow = ctx.getConfiguration().getInt(MAX_SIMILARITIES_PER_ROW, 0);
      Preconditions.checkArgument(maxSimilaritiesPerRow > 0, "Maximum number of similarities per row must be greater then 0!");
    }

    @Override
    protected void reduce(IntWritable row, Iterable<VectorWritable> partials, Context ctx)
      throws IOException, InterruptedException {
      Vector allSimilarities = Vectors.merge(partials);
      Vector topKSimilarities = Vectors.topKElements(maxSimilaritiesPerRow, allSimilarities);
      ctx.write(row, new VectorWritable(topKSimilarities));
    }
  }

}
