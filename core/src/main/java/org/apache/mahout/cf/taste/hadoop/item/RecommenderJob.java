/*
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

package org.apache.mahout.cf.taste.hadoop.item;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.OutputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.RecommendedItemsWritable;
import org.apache.mahout.cf.taste.hadoop.preparation.PreparePreferenceMatrixJob;
import org.apache.mahout.cf.taste.hadoop.similarity.item.ItemSimilarityJob;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * <p>Runs a completely distributed recommender job as a series of mapreduces.</p>
 * <p/>
 * <p>Preferences in the input file should look like {@code userID, itemID[, preferencevalue]}</p>
 * <p/>
 * <p>
 * Preference value is optional to accommodate applications that have no notion of a preference value (that is, the user
 * simply expresses a preference for an item, but no degree of preference).
 * </p>
 * <p/>
 * <p>
 * The preference value is assumed to be parseable as a {@code double}. The user IDs and item IDs are
 * parsed as {@code long}s.
 * </p>
 * <p/>
 * <p>Command line arguments specific to this class are:</p>
 * <p/>
 * <ol>
 * <li>--input(path): Directory containing one or more text files with the preference data</li>
 * <li>--output(path): output path where recommender output should go</li>
 * <li>--similarityClassname (classname): Name of vector similarity class to instantiate or a predefined similarity
 * from {@link org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasure}</li>
 * <li>--usersFile (path): only compute recommendations for user IDs contained in this file (optional)</li>
 * <li>--itemsFile (path): only include item IDs from this file in the recommendations (optional)</li>
 * <li>--filterFile (path): file containing comma-separated userID,itemID pairs. Used to exclude the item from the
 * recommendations for that user (optional)</li>
 * <li>--numRecommendations (integer): Number of recommendations to compute per user (10)</li>
 * <li>--booleanData (boolean): Treat input data as having no pref values (false)</li>
 * <li>--maxPrefsPerUser (integer): Maximum number of preferences considered per user in final
 *   recommendation phase (10)</li>
 * <li>--maxSimilaritiesPerItem (integer): Maximum number of similarities considered per item (100)</li>
 * <li>--minPrefsPerUser (integer): ignore users with less preferences than this in the similarity computation (1)</li>
 * <li>--maxPrefsPerUserInItemSimilarity (integer): max number of preferences to consider per user in
 *   the item similarity computation phase,
 * users with more preferences will be sampled down (1000)</li>
 * <li>--threshold (double): discard item pairs with a similarity value below this</li>
 * </ol>
 * <p/>
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 * <p/>
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other
 * arguments.</p>
 */
public final class RecommenderJob extends AbstractJob {

  public static final String BOOLEAN_DATA = "booleanData";
  public static final String DEFAULT_PREPARE_PATH = "preparePreferenceMatrix";

  private static final int DEFAULT_MAX_SIMILARITIES_PER_ITEM = 100;
  private static final int DEFAULT_MAX_PREFS = 500;
  private static final int DEFAULT_MIN_PREFS_PER_USER = 1;

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("numRecommendations", "n", "Number of recommendations per user",
            String.valueOf(AggregateAndRecommendReducer.DEFAULT_NUM_RECOMMENDATIONS));
    addOption("usersFile", null, "File of users to recommend for", null);
    addOption("itemsFile", null, "File of items to recommend for", null);
    addOption("filterFile", "f", "File containing comma-separated userID,itemID pairs. Used to exclude the item from "
            + "the recommendations for that user (optional)", null);
    addOption("booleanData", "b", "Treat input as without pref values", Boolean.FALSE.toString());
    addOption("maxPrefsPerUser", "mxp",
            "Maximum number of preferences considered per user in final recommendation phase",
            String.valueOf(UserVectorSplitterMapper.DEFAULT_MAX_PREFS_PER_USER_CONSIDERED));
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this in the similarity computation "
            + "(default: " + DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(DEFAULT_MIN_PREFS_PER_USER));
    addOption("maxSimilaritiesPerItem", "m", "Maximum number of similarities considered per item ",
            String.valueOf(DEFAULT_MAX_SIMILARITIES_PER_ITEM));
    addOption("maxPrefsInItemSimilarity", "mpiis", "max number of preferences to consider per user or item in the "
            + "item similarity computation phase, users or items with more preferences will be sampled down (default: "
        + DEFAULT_MAX_PREFS + ')', String.valueOf(DEFAULT_MAX_PREFS));
    addOption("similarityClassname", "s", "Name of distributed similarity measures class to instantiate, " 
            + "alternatively use one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')', true);
    addOption("threshold", "tr", "discard item pairs with a similarity value below this", false);
    addOption("outputPathForSimilarityMatrix", "opfsm", "write the item similarity matrix to this path (optional)",
        false);
    addOption("randomSeed", null, "use this seed for sampling", false);
    addFlag("sequencefileOutput", null, "write the output into a SequenceFile instead of a text file");

    Map<String, List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path outputPath = getOutputPath();
    int numRecommendations = Integer.parseInt(getOption("numRecommendations"));
    String usersFile = getOption("usersFile");
    String itemsFile = getOption("itemsFile");
    String filterFile = getOption("filterFile");
    boolean booleanData = Boolean.valueOf(getOption("booleanData"));
    int maxPrefsPerUser = Integer.parseInt(getOption("maxPrefsPerUser"));
    int minPrefsPerUser = Integer.parseInt(getOption("minPrefsPerUser"));
    int maxPrefsInItemSimilarity = Integer.parseInt(getOption("maxPrefsInItemSimilarity"));
    int maxSimilaritiesPerItem = Integer.parseInt(getOption("maxSimilaritiesPerItem"));
    String similarityClassname = getOption("similarityClassname");
    double threshold = hasOption("threshold")
        ? Double.parseDouble(getOption("threshold")) : RowSimilarityJob.NO_THRESHOLD;
    long randomSeed = hasOption("randomSeed")
        ? Long.parseLong(getOption("randomSeed")) : RowSimilarityJob.NO_FIXED_RANDOM_SEED;


    Path prepPath = getTempPath(DEFAULT_PREPARE_PATH);
    Path similarityMatrixPath = getTempPath("similarityMatrix");
    Path explicitFilterPath = getTempPath("explicitFilterPath");
    Path partialMultiplyPath = getTempPath("partialMultiply");

    AtomicInteger currentPhase = new AtomicInteger();

    int numberOfUsers = -1;

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      ToolRunner.run(getConf(), new PreparePreferenceMatrixJob(), new String[]{
        "--input", getInputPath().toString(),
        "--output", prepPath.toString(),
        "--minPrefsPerUser", String.valueOf(minPrefsPerUser),
        "--booleanData", String.valueOf(booleanData),
        "--tempDir", getTempPath().toString(),
      });

      numberOfUsers = HadoopUtil.readInt(new Path(prepPath, PreparePreferenceMatrixJob.NUM_USERS), getConf());
    }


    if (shouldRunNextPhase(parsedArgs, currentPhase)) {

      /* special behavior if phase 1 is skipped */
      if (numberOfUsers == -1) {
        numberOfUsers = (int) HadoopUtil.countRecords(new Path(prepPath, PreparePreferenceMatrixJob.USER_VECTORS),
                PathType.LIST, null, getConf());
      }

      //calculate the co-occurrence matrix
      ToolRunner.run(getConf(), new RowSimilarityJob(), new String[]{
        "--input", new Path(prepPath, PreparePreferenceMatrixJob.RATING_MATRIX).toString(),
        "--output", similarityMatrixPath.toString(),
        "--numberOfColumns", String.valueOf(numberOfUsers),
        "--similarityClassname", similarityClassname,
        "--maxObservationsPerRow", String.valueOf(maxPrefsInItemSimilarity),
        "--maxObservationsPerColumn", String.valueOf(maxPrefsInItemSimilarity),
        "--maxSimilaritiesPerRow", String.valueOf(maxSimilaritiesPerItem),
        "--excludeSelfSimilarity", String.valueOf(Boolean.TRUE),
        "--threshold", String.valueOf(threshold),
        "--randomSeed", String.valueOf(randomSeed),
        "--tempDir", getTempPath().toString(),
      });

      // write out the similarity matrix if the user specified that behavior
      if (hasOption("outputPathForSimilarityMatrix")) {
        Path outputPathForSimilarityMatrix = new Path(getOption("outputPathForSimilarityMatrix"));

        Job outputSimilarityMatrix = prepareJob(similarityMatrixPath, outputPathForSimilarityMatrix,
            SequenceFileInputFormat.class, ItemSimilarityJob.MostSimilarItemPairsMapper.class,
            EntityEntityWritable.class, DoubleWritable.class, ItemSimilarityJob.MostSimilarItemPairsReducer.class,
            EntityEntityWritable.class, DoubleWritable.class, TextOutputFormat.class);

        Configuration mostSimilarItemsConf = outputSimilarityMatrix.getConfiguration();
        mostSimilarItemsConf.set(ItemSimilarityJob.ITEM_ID_INDEX_PATH_STR,
            new Path(prepPath, PreparePreferenceMatrixJob.ITEMID_INDEX).toString());
        mostSimilarItemsConf.setInt(ItemSimilarityJob.MAX_SIMILARITIES_PER_ITEM, maxSimilaritiesPerItem);
        outputSimilarityMatrix.waitForCompletion(true);
      }
    }

    //start the multiplication of the co-occurrence matrix by the user vectors
    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job partialMultiply = new Job(getConf(), "partialMultiply");
      Configuration partialMultiplyConf = partialMultiply.getConfiguration();

      MultipleInputs.addInputPath(partialMultiply, similarityMatrixPath, SequenceFileInputFormat.class,
                                  SimilarityMatrixRowWrapperMapper.class);
      MultipleInputs.addInputPath(partialMultiply, new Path(prepPath, PreparePreferenceMatrixJob.USER_VECTORS),
          SequenceFileInputFormat.class, UserVectorSplitterMapper.class);
      partialMultiply.setJarByClass(ToVectorAndPrefReducer.class);
      partialMultiply.setMapOutputKeyClass(VarIntWritable.class);
      partialMultiply.setMapOutputValueClass(VectorOrPrefWritable.class);
      partialMultiply.setReducerClass(ToVectorAndPrefReducer.class);
      partialMultiply.setOutputFormatClass(SequenceFileOutputFormat.class);
      partialMultiply.setOutputKeyClass(VarIntWritable.class);
      partialMultiply.setOutputValueClass(VectorAndPrefsWritable.class);
      partialMultiplyConf.setBoolean("mapred.compress.map.output", true);
      partialMultiplyConf.set("mapred.output.dir", partialMultiplyPath.toString());

      if (usersFile != null) {
        partialMultiplyConf.set(UserVectorSplitterMapper.USERS_FILE, usersFile);
      }
      partialMultiplyConf.setInt(UserVectorSplitterMapper.MAX_PREFS_PER_USER_CONSIDERED, maxPrefsPerUser);

      boolean succeeded = partialMultiply.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      //filter out any users we don't care about
      /* convert the user/item pairs to filter if a filterfile has been specified */
      if (filterFile != null) {
        Job itemFiltering = prepareJob(new Path(filterFile), explicitFilterPath, TextInputFormat.class,
                ItemFilterMapper.class, VarLongWritable.class, VarLongWritable.class,
                ItemFilterAsVectorAndPrefsReducer.class, VarIntWritable.class, VectorAndPrefsWritable.class,
                SequenceFileOutputFormat.class);
        boolean succeeded = itemFiltering.waitForCompletion(true);
        if (!succeeded) {
          return -1;
        }
      }

      String aggregateAndRecommendInput = partialMultiplyPath.toString();
      if (filterFile != null) {
        aggregateAndRecommendInput += "," + explicitFilterPath;
      }

      Class<? extends OutputFormat> outputFormat = parsedArgs.containsKey("--sequencefileOutput")
          ? SequenceFileOutputFormat.class : TextOutputFormat.class;

      //extract out the recommendations
      Job aggregateAndRecommend = prepareJob(
              new Path(aggregateAndRecommendInput), outputPath, SequenceFileInputFormat.class,
              PartialMultiplyMapper.class, VarLongWritable.class, PrefAndSimilarityColumnWritable.class,
              AggregateAndRecommendReducer.class, VarLongWritable.class, RecommendedItemsWritable.class,
              outputFormat);
      Configuration aggregateAndRecommendConf = aggregateAndRecommend.getConfiguration();
      if (itemsFile != null) {
        aggregateAndRecommendConf.set(AggregateAndRecommendReducer.ITEMS_FILE, itemsFile);
      }

      if (filterFile != null) {
        setS3SafeCombinedInputPath(aggregateAndRecommend, getTempPath(), partialMultiplyPath, explicitFilterPath);
      }
      setIOSort(aggregateAndRecommend);
      aggregateAndRecommendConf.set(AggregateAndRecommendReducer.ITEMID_INDEX_PATH,
              new Path(prepPath, PreparePreferenceMatrixJob.ITEMID_INDEX).toString());
      aggregateAndRecommendConf.setInt(AggregateAndRecommendReducer.NUM_RECOMMENDATIONS, numRecommendations);
      aggregateAndRecommendConf.setBoolean(BOOLEAN_DATA, booleanData);
      boolean succeeded = aggregateAndRecommend.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
    }

    return 0;
  }

  private static void setIOSort(JobContext job) {
    Configuration conf = job.getConfiguration();
    conf.setInt("io.sort.factor", 100);
    String javaOpts = conf.get("mapred.map.child.java.opts"); // new arg name
    if (javaOpts == null) {
      javaOpts = conf.get("mapred.child.java.opts"); // old arg name
    }
    int assumedHeapSize = 512;
    if (javaOpts != null) {
      Matcher m = Pattern.compile("-Xmx([0-9]+)([mMgG])").matcher(javaOpts);
      if (m.find()) {
        assumedHeapSize = Integer.parseInt(m.group(1));
        String megabyteOrGigabyte = m.group(2);
        if ("g".equalsIgnoreCase(megabyteOrGigabyte)) {
          assumedHeapSize *= 1024;
        }
      }
    }
    // Cap this at 1024MB now; see https://issues.apache.org/jira/browse/MAPREDUCE-2308
    conf.setInt("io.sort.mb", Math.min(assumedHeapSize / 2, 1024));
    // For some reason the Merger doesn't report status for a long time; increase
    // timeout when running these jobs
    conf.setInt("mapred.task.timeout", 60 * 60 * 1000);
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new RecommenderJob(), args);
  }
}
