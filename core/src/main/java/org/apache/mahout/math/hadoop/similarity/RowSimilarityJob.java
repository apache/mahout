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

package org.apache.mahout.math.hadoop.similarity;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;
import org.apache.mahout.math.hadoop.DistributedRowMatrix.MatrixEntryWritable;
import org.apache.mahout.math.hadoop.similarity.vector.DistributedVectorSimilarity;

/**
 * <p>Runs a completely distributed computation of the pairwise similarity of the row vectors of a
 * {@link DistributedRowMatrix} as a series of mapreduces.</p>
 *
 * <p>The algorithm used is a slight modification of the algorithm described in
 * T. Elsayed et al: &quot;Pairwise document similarity in large collections with MapReduce&quot; 
 * (http://www.umiacs.umd.edu/~jimmylin/publications/Elsayed_etal_ACL2008_short.pdf)</p>
 *
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>-Dmapred.input.dir=(path): Directory containing a {@link DistributedRowMatrix} as a
 * SequenceFile<IntWritable,VectorWritable></li>
 * <li>-Dmapred.output.dir=(path): output path where the computations output should go (a {@link DistributedRowMatrix}
 * stored as a SequenceFile<IntWritable,VectorWritable>)</li>
 * <li>--numberOfColumns: the number of columns in the input matrix</li>
 * <li>--similarityClassname (classname): an implementation of {@link DistributedVectorSimilarity} used to compute the
 * similarity</li>
 * <li>--maxSimilaritiesPerRow (integer): cap the number of similar rows per row to this number (100)</li>
 * </ol>
 *
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 * <p>Please consider supplying a --tempDir parameter for this job, as is needs to write some intermediate files</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public class RowSimilarityJob extends AbstractJob {

  public static final String DISTRIBUTED_SIMILARITY_CLASSNAME =
      RowSimilarityJob.class.getName() + ".distributedSimilarityClassname";
  public static final String NUMBER_OF_COLUMNS = RowSimilarityJob.class.getName() + ".numberOfColumns";
  public static final String MAX_SIMILARITIES_PER_ROW = RowSimilarityJob.class.getName() + ".maxSimilaritiesPerRow";

  private static final int DEFAULT_MAX_SIMILARITIES_PER_ROW = 100;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new RowSimilarityJob(), args);
  }

  @Override
  public int run(String[] args) throws IOException, ClassNotFoundException, InterruptedException {

    addInputOption();
    addOutputOption();
    addOption("numberOfColumns", "r", "Number of columns in the input matrix");
    addOption("similarityClassname", "s", "Name of distributed similarity class to instantiate, alternatively use " + 
        "one of the predefined similarities (" + SimilarityType.listEnumNames() + ')');
    addOption("maxSimilaritiesPerRow", "m", "Number of maximum similarities per row (default: "
              + DEFAULT_MAX_SIMILARITIES_PER_ROW + ')', String.valueOf(DEFAULT_MAX_SIMILARITIES_PER_ROW));

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    int numberOfColumns = Integer.parseInt(parsedArgs.get("--numberOfColumns"));
    String similarityClassnameArg = parsedArgs.get("--similarityClassname");
    String distributedSimilarityClassname;
    try {
      distributedSimilarityClassname =
          SimilarityType.valueOf(similarityClassnameArg).getSimilarityImplementationClassName();
    } catch (IllegalArgumentException iae) {
      distributedSimilarityClassname = similarityClassnameArg;
    }

    int maxSimilaritiesPerRow = Integer.parseInt(parsedArgs.get("--maxSimilaritiesPerRow"));

    Path inputPath = getInputPath();
    Path outputPath = getOutputPath();
    Path tempDirPath = new Path(parsedArgs.get("--tempDir"));

    Path weightsPath = new Path(tempDirPath, "weights");
    Path pairwiseSimilarityPath = new Path(tempDirPath, "pairwiseSimilarity");

    AtomicInteger currentPhase = new AtomicInteger();

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job weights = prepareJob(inputPath,
                               weightsPath,
                               SequenceFileInputFormat.class,
                               RowWeightMapper.class,
                               VarIntWritable.class,
                               WeightedOccurrence.class,
                               WeightedOccurrencesPerColumnReducer.class,
                               VarIntWritable.class,
                               WeightedOccurrenceArray.class,
                               SequenceFileOutputFormat.class);

      weights.getConfiguration().set(DISTRIBUTED_SIMILARITY_CLASSNAME, distributedSimilarityClassname);
      weights.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job pairwiseSimilarity = prepareJob(weightsPath,
                               pairwiseSimilarityPath,
                               SequenceFileInputFormat.class,
                               CooccurrencesMapper.class,
                               WeightedRowPair.class,
                               Cooccurrence.class,
                               SimilarityReducer.class,
                               SimilarityMatrixEntryKey.class,
                               MatrixEntryWritable.class,
                               SequenceFileOutputFormat.class);

      Configuration pairwiseConf = pairwiseSimilarity.getConfiguration();
      pairwiseConf.set(DISTRIBUTED_SIMILARITY_CLASSNAME, distributedSimilarityClassname);
      pairwiseConf.setInt(NUMBER_OF_COLUMNS, numberOfColumns);
      pairwiseSimilarity.waitForCompletion(true);
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job asMatrix = prepareJob(pairwiseSimilarityPath,
                               outputPath,
                               SequenceFileInputFormat.class,
                               Mapper.class,
                               SimilarityMatrixEntryKey.class,
                               MatrixEntryWritable.class,
                               EntriesToVectorsReducer.class,
                               IntWritable.class,
                               VectorWritable.class,
                               SequenceFileOutputFormat.class);
      asMatrix.setPartitionerClass(HashPartitioner.class);
      asMatrix.setGroupingComparatorClass(SimilarityMatrixEntryKey.SimilarityMatrixEntryKeyGroupingComparator.class);
      asMatrix.getConfiguration().setInt(MAX_SIMILARITIES_PER_ROW, maxSimilaritiesPerRow);
      asMatrix.waitForCompletion(true);
    }

    return 0;
  }

  static DistributedVectorSimilarity instantiateSimilarity(String classname) {
    try {
      return Class.forName(classname).asSubclass(DistributedVectorSimilarity.class).newInstance();
    } catch (ClassNotFoundException cnfe) {
      throw new IllegalStateException(cnfe);
    } catch (InstantiationException ie) {
      throw new IllegalStateException(ie);
    } catch (IllegalAccessException iae) {
      throw new IllegalStateException(iae);
    }
  }

  /**
   * applies {@link DistributedVectorSimilarity#weight(Vector)} to each row of the input matrix
   */
  public static class RowWeightMapper extends Mapper<IntWritable,VectorWritable,VarIntWritable,WeightedOccurrence> {

    private DistributedVectorSimilarity similarity;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      super.setup(ctx);
      similarity = instantiateSimilarity(ctx.getConfiguration().get(DISTRIBUTED_SIMILARITY_CLASSNAME));
    }

    @Override
    protected void map(IntWritable row, VectorWritable vectorWritable, Context ctx)
        throws IOException, InterruptedException {

      Vector v = vectorWritable.get();
      double weight = similarity.weight(v);

      Iterator<Vector.Element> elementsIterator = v.iterateNonZero();
      while (elementsIterator.hasNext()) {
        Vector.Element element = elementsIterator.next();
        int column = element.index();
        double value = element.get();
        ctx.write(new VarIntWritable(column), new WeightedOccurrence(row.get(), value, weight));
      }
    }
  }

  /**
   * collects all {@link WeightedOccurrence}s for a column and writes them to a {@link WeightedOccurrenceArray}
   */
  public static class WeightedOccurrencesPerColumnReducer
      extends Reducer<VarIntWritable,WeightedOccurrence,VarIntWritable,WeightedOccurrenceArray> {

    @Override
    protected void reduce(VarIntWritable column, Iterable<WeightedOccurrence> weightedOccurrences, Context ctx)
        throws IOException, InterruptedException {

      List<WeightedOccurrence> collectedWeightedOccurrences = new ArrayList<WeightedOccurrence>();
      for (WeightedOccurrence weightedOccurrence : weightedOccurrences) {
        collectedWeightedOccurrences.add(weightedOccurrence.clone());
      }

      ctx.write(column, new WeightedOccurrenceArray(collectedWeightedOccurrences.toArray(
          new WeightedOccurrence[collectedWeightedOccurrences.size()])));
    }
  }

  /**
   * maps all pairs of weighted entries of a column vector
   */
  public static class CooccurrencesMapper
      extends Mapper<VarIntWritable,WeightedOccurrenceArray,WeightedRowPair,Cooccurrence> {

    @Override
    protected void map(VarIntWritable column, WeightedOccurrenceArray weightedOccurrenceArray, Context ctx)
        throws IOException, InterruptedException {

      WeightedOccurrence[] weightedOccurrences = weightedOccurrenceArray.getWeightedOccurrences();

      WeightedRowPair rowPair = new WeightedRowPair();
      Cooccurrence coocurrence = new Cooccurrence();

      for (int n = 0; n < weightedOccurrences.length; n++) {
        int rowA = weightedOccurrences[n].getRow();
        double weightA = weightedOccurrences[n].getWeight();
        double valueA = weightedOccurrences[n].getValue();
        for (int m = n; m < weightedOccurrences.length; m++) {
          int rowB = weightedOccurrences[m].getRow();
          double weightB = weightedOccurrences[m].getWeight();
          double valueB = weightedOccurrences[m].getValue();
          rowPair.set(rowA, rowB, weightA, weightB);
          coocurrence.set(column.get(), valueA, valueB);
          ctx.write(rowPair, coocurrence);
        }
      }
    }
  }

  /**
   * computes the pairwise similarities
   */
  public static class SimilarityReducer
      extends Reducer<WeightedRowPair,Cooccurrence,SimilarityMatrixEntryKey, MatrixEntryWritable> {

    private DistributedVectorSimilarity similarity;
    private int numberOfColumns;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      super.setup(ctx);
      similarity = instantiateSimilarity(ctx.getConfiguration().get(DISTRIBUTED_SIMILARITY_CLASSNAME));
      numberOfColumns = ctx.getConfiguration().getInt(NUMBER_OF_COLUMNS, -1);
      if (numberOfColumns < 1) {
        throw new IllegalStateException("Number of columns was not correctly set!");
      }
    }

    @Override
    protected void reduce(WeightedRowPair rowPair, Iterable<Cooccurrence> cooccurrences, Context ctx)
        throws IOException, InterruptedException {

      int rowA = rowPair.getRowA();
      int rowB = rowPair.getRowB();
      double similarityValue = similarity.similarity(rowA, rowB, cooccurrences, rowPair.getWeightA(),
          rowPair.getWeightB(), numberOfColumns);

      if (!Double.isNaN(similarityValue)) {
        SimilarityMatrixEntryKey key = new SimilarityMatrixEntryKey();
        MatrixEntryWritable entry = new MatrixEntryWritable();
        entry.setVal(similarityValue);

        entry.setRow(rowA);
        entry.setCol(rowB);
        key.set(rowA, similarityValue);
        ctx.write(key, entry);

        if (rowA != rowB) {
          entry.setRow(rowB);
          entry.setCol(rowA);
          key.set(rowB, similarityValue);
          ctx.write(key, entry);
        }
      }
    }
  }

  /**
   * collects all {@link MatrixEntryWritable} for each column and creates a {@link VectorWritable}
   */
  public static class EntriesToVectorsReducer
      extends Reducer<SimilarityMatrixEntryKey, MatrixEntryWritable,IntWritable,VectorWritable> {

    private int maxSimilaritiesPerRow;

    @Override
    protected void setup(Context ctx) throws IOException, InterruptedException {
      super.setup(ctx);
      maxSimilaritiesPerRow = ctx.getConfiguration().getInt(MAX_SIMILARITIES_PER_ROW, -1);
      if (maxSimilaritiesPerRow < 1) {
        throw new IllegalStateException("Maximum number of similarities per row was not correctly set!");
      }
    }

    @Override
    protected void reduce(SimilarityMatrixEntryKey key, Iterable<MatrixEntryWritable> entries, Context ctx)
        throws IOException, InterruptedException {
      RandomAccessSparseVector temporaryVector = new RandomAccessSparseVector(Integer.MAX_VALUE, maxSimilaritiesPerRow);
      int similaritiesSet = 0;
      for (MatrixEntryWritable entry : entries) {
        temporaryVector.setQuick(entry.getCol(), entry.getVal());
        if (++similaritiesSet == maxSimilaritiesPerRow) {
          break;
        }
      }
      SequentialAccessSparseVector vector = new SequentialAccessSparseVector(temporaryVector);
      ctx.write(new IntWritable(key.getRow()), new VectorWritable(vector));
    }
  }

}
