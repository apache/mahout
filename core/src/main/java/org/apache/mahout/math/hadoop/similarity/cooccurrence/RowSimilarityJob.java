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
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.common.TopK;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
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
import java.util.concurrent.atomic.AtomicInteger;

public class RowSimilarityJob extends AbstractJob {

  public static final double NO_THRESHOLD = Double.MIN_VALUE;

  static final String SIMILARITY_CLASSNAME = RowSimilarityJob.class + ".distributedSimilarityClassname";
  static final String NUMBER_OF_COLUMNS = RowSimilarityJob.class + ".numberOfColumns";
  static final String MAX_SIMILARITIES_PER_ROW = RowSimilarityJob.class + ".maxSimilaritiesPerRow";
  static final String EXCLUDE_SELF_SIMILARITY = RowSimilarityJob.class + ".excludeSelfSimilarity";

  static final String THRESHOLD = RowSimilarityJob.class + ".threshold";
  static final String NORMS_PATH = RowSimilarityJob.class + ".normsPath";
  static final String MAXVALUES_PATH = RowSimilarityJob.class + ".maxWeightsPath";

  static final String NUM_NON_ZERO_ENTRIES_PATH = RowSimilarityJob.class + ".nonZeroEntriesPath";
  private static final int DEFAULT_MAX_SIMILARITIES_PER_ROW = 100;

  private static final int NORM_VECTOR_MARKER = Integer.MIN_VALUE;
  private static final int MAXVALUE_VECTOR_MARKER = Integer.MIN_VALUE + 1;
  private static final int NUM_NON_ZERO_ENTRIES_VECTOR_MARKER = Integer.MIN_VALUE + 2;

  enum Counters { ROWS, COOCCURRENCES, PRUNED_COOCCURRENCES }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RowSimilarityJob(), args);
  }


  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("numberOfColumns", "r", "Number of columns in the input matrix");
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate, alternatively use "
        + "one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')');
    addOption("maxSimilaritiesPerRow", "m", "Number of maximum similarities per row (default: "
        + DEFAULT_MAX_SIMILARITIES_PER_ROW + ')', String.valueOf(DEFAULT_MAX_SIMILARITIES_PER_ROW));
    addOption("excludeSelfSimilarity", "ess", "compute similarity of rows to themselves?", String.valueOf(false));
    addOption("threshold", "tr", "discard row pairs with a similarity value below this", false);

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    int numberOfColumns = Integer.parseInt(getOption("numberOfColumns"));
    String similarityClassnameArg = getOption("similarityClassname");
    String similarityClassname;
    try {
      similarityClassname = VectorSimilarityMeasures.valueOf(similarityClassnameArg).getClassname();
    } catch (IllegalArgumentException iae) {
      similarityClassname = similarityClassnameArg;
    }

    int maxSimilaritiesPerRow = Integer.parseInt(getOption("maxSimilaritiesPerRow"));
    boolean excludeSelfSimilarity = Boolean.parseBoolean(getOption("excludeSelfSimilarity"));
    double threshold = hasOption("threshold") ?
        Double.parseDouble(getOption("threshold")) : NO_THRESHOLD;

    Path weightsPath = getTempPath("weights");
    Path normsPath = getTempPath("norms.bin");
    Path numNonZeroEntriesPath = getTempPath("numNonZeroEntries.bin");
    Path maxValuesPath = getTempPath("maxValues.bin");
    Path pairwiseSimilarityPath = getTempPath("pairwiseSimilarity");

    AtomicInteger currentPhase = new AtomicInteger();

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

  public static class VectorNormMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private VectorSimilarityMeasure similarity;
    private Vector norms;
    private Vector nonZeroEntries;
    private Vector maxValues;
    private double threshold;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      similarity = ClassUtils.instantiateAs(ctx.getConfiguration().get(SIMILARITY_CLASSNAME),
          VectorSimilarityMeasure.class);
      norms = new RandomAccessSparseVector(Integer.MAX_VALUE);
      nonZeroEntries = new RandomAccessSparseVector(Integer.MAX_VALUE);
      maxValues = new RandomAccessSparseVector(Integer.MAX_VALUE);
      threshold = Double.parseDouble(ctx.getConfiguration().get(THRESHOLD));
    }

    @Override
    protected void map(IntWritable row, VectorWritable vectorWritable, Context ctx)
        throws IOException, InterruptedException {

      Vector rowVector = similarity.normalize(vectorWritable.get());

      int numNonZeroEntries = 0;
      double maxValue = Double.MIN_VALUE;

      Iterator<Vector.Element> nonZeroElements = rowVector.iterateNonZero();
      while (nonZeroElements.hasNext()) {
        Vector.Element element = nonZeroElements.next();
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
      super.cleanup(ctx);
      // dirty trick
      ctx.write(new IntWritable(NORM_VECTOR_MARKER), new VectorWritable(norms));
      ctx.write(new IntWritable(NUM_NON_ZERO_ENTRIES_VECTOR_MARKER), new VectorWritable(nonZeroEntries));
      ctx.write(new IntWritable(MAXVALUE_VECTOR_MARKER), new VectorWritable(maxValues));
    }
  }

  public static class MergeVectorsCombiner extends Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {
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


  public static class SimilarityReducer
      extends Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {

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
      Preconditions.checkArgument(numberOfColumns > 0, "Incorrect number of columns!");
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
        Iterator<Vector.Element> nonZeroElements = toAdd.iterateNonZero();
        while (nonZeroElements.hasNext()) {
          Vector.Element nonZeroElement = nonZeroElements.next();
          dots.setQuick(nonZeroElement.index(), dots.getQuick(nonZeroElement.index()) + nonZeroElement.get());
        }
      }

      Vector similarities = dots.like();
      double normA = norms.getQuick(row.get());
      Iterator<Vector.Element> dotsWith = dots.iterateNonZero();
      while (dotsWith.hasNext()) {
        Vector.Element b = dotsWith.next();
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
      Preconditions.checkArgument(maxSimilaritiesPerRow > 0, "Incorrect maximum number of similarities per row!");
    }

    @Override
    protected void map(IntWritable row, VectorWritable similaritiesWritable, Context ctx)
        throws IOException, InterruptedException {
      Vector similarities = similaritiesWritable.get();
      // For performance reasons moved transposedPartial creation out of the while loop and reusing the same vector
      Vector transposedPartial = similarities.like();
      TopK<Vector.Element> topKQueue = new TopK<Vector.Element>(maxSimilaritiesPerRow, Vectors.BY_VALUE);
      Iterator<Vector.Element> nonZeroElements = similarities.iterateNonZero();
      while (nonZeroElements.hasNext()) {
        Vector.Element nonZeroElement = nonZeroElements.next();
        topKQueue.offer(new Vectors.TemporaryElement(nonZeroElement));
        transposedPartial.setQuick(row.get(), nonZeroElement.get());
        ctx.write(new IntWritable(nonZeroElement.index()), new VectorWritable(transposedPartial));
        transposedPartial.setQuick(row.get(), 0.0);
      }
      Vector topKSimilarities = similarities.like();
      for (Vector.Element topKSimilarity : topKQueue.retrieve()) {
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
      Preconditions.checkArgument(maxSimilaritiesPerRow > 0, "Incorrect maximum number of similarities per row!");
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
