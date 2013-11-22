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

package org.apache.mahout.cf.taste.hadoop.similarity.item;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.EntityEntityWritable;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.hadoop.preparation.PreparePreferenceMatrixJob;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItem;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.RowSimilarityJob;
import org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasures;
import org.apache.mahout.math.map.OpenIntLongHashMap;

/**
 * <p>Distributed precomputation of the item-item-similarities for Itembased Collaborative Filtering</p>
 *
 * <p>Preferences in the input file should look like {@code userID,itemID[,preferencevalue]}</p>
 *
 * <p>
 * Preference value is optional to accommodate applications that have no notion of a preference value (that is, the user
 * simply expresses a preference for an item, but no degree of preference).
 * </p>
 *
 * <p>
 * The preference value is assumed to be parseable as a {@code double}. The user IDs and item IDs are
 * parsed as {@code long}s.
 * </p>
 *
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--input (path): Directory containing one or more text files with the preference data</li>
 * <li>--output (path): output path where similarity data should be written</li>
 * <li>--similarityClassname (classname): Name of distributed similarity measure class to instantiate or a predefined
 *  similarity from {@link org.apache.mahout.math.hadoop.similarity.cooccurrence.measures.VectorSimilarityMeasure}</li>
 * <li>--maxSimilaritiesPerItem (integer): Maximum number of similarities considered per item (100)</li>
 * <li>--maxPrefsPerUser (integer): max number of preferences to consider per user, users with more preferences will
 *  be sampled down (1000)</li>
 * <li>--minPrefsPerUser (integer): ignore users with less preferences than this (1)</li>
 * <li>--booleanData (boolean): Treat input data as having no pref values (false)</li>
 * <li>--threshold (double): discard item pairs with a similarity value below this</li>
 * </ol>
 *
 * <p>General command line options are documented in {@link AbstractJob}.</p>
 *
 * <p>Note that because of how Hadoop parses arguments, all "-D" arguments must appear before all other arguments.</p>
 */
public final class ItemSimilarityJob extends AbstractJob {

  public static final String ITEM_ID_INDEX_PATH_STR = ItemSimilarityJob.class.getName() + ".itemIDIndexPathStr";
  public static final String MAX_SIMILARITIES_PER_ITEM = ItemSimilarityJob.class.getName() + ".maxSimilarItemsPerItem";

  private static final int DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM = 100;
  private static final int DEFAULT_MAX_PREFS = 500;
  private static final int DEFAULT_MIN_PREFS_PER_USER = 1;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ItemSimilarityJob(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("similarityClassname", "s", "Name of distributed similarity measures class to instantiate, " 
        + "alternatively use one of the predefined similarities (" + VectorSimilarityMeasures.list() + ')');
    addOption("maxSimilaritiesPerItem", "m", "try to cap the number of similar items per item to this number "
        + "(default: " + DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM + ')',
        String.valueOf(DEFAULT_MAX_SIMILAR_ITEMS_PER_ITEM));
    addOption("maxPrefs", "mppu", "max number of preferences to consider per user or item, " 
        + "users or items with more preferences will be sampled down (default: " + DEFAULT_MAX_PREFS + ')',
        String.valueOf(DEFAULT_MAX_PREFS));
    addOption("minPrefsPerUser", "mp", "ignore users with less preferences than this "
        + "(default: " + DEFAULT_MIN_PREFS_PER_USER + ')', String.valueOf(DEFAULT_MIN_PREFS_PER_USER));
    addOption("booleanData", "b", "Treat input as without pref values", String.valueOf(Boolean.FALSE));
    addOption("threshold", "tr", "discard item pairs with a similarity value below this", false);
    addOption("randomSeed", null, "use this seed for sampling", false);

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    String similarityClassName = getOption("similarityClassname");
    int maxSimilarItemsPerItem = Integer.parseInt(getOption("maxSimilaritiesPerItem"));
    int maxPrefs = Integer.parseInt(getOption("maxPrefs"));
    int minPrefsPerUser = Integer.parseInt(getOption("minPrefsPerUser"));
    boolean booleanData = Boolean.valueOf(getOption("booleanData"));

    double threshold = hasOption("threshold")
        ? Double.parseDouble(getOption("threshold")) : RowSimilarityJob.NO_THRESHOLD;
    long randomSeed = hasOption("randomSeed")
        ? Long.parseLong(getOption("randomSeed")) : RowSimilarityJob.NO_FIXED_RANDOM_SEED;

    Path similarityMatrixPath = getTempPath("similarityMatrix");
    Path prepPath = getTempPath("prepareRatingMatrix");

    AtomicInteger currentPhase = new AtomicInteger();

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      ToolRunner.run(getConf(), new PreparePreferenceMatrixJob(), new String[] {
        "--input", getInputPath().toString(),
        "--output", prepPath.toString(),
        "--minPrefsPerUser", String.valueOf(minPrefsPerUser),
        "--booleanData", String.valueOf(booleanData),
        "--tempDir", getTempPath().toString(),
      });
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      int numberOfUsers = HadoopUtil.readInt(new Path(prepPath, PreparePreferenceMatrixJob.NUM_USERS), getConf());

      ToolRunner.run(getConf(), new RowSimilarityJob(), new String[] {
        "--input", new Path(prepPath, PreparePreferenceMatrixJob.RATING_MATRIX).toString(),
        "--output", similarityMatrixPath.toString(),
        "--numberOfColumns", String.valueOf(numberOfUsers),
        "--similarityClassname", similarityClassName,
        "--maxObservationsPerRow", String.valueOf(maxPrefs),
        "--maxObservationsPerColumn", String.valueOf(maxPrefs),
        "--maxSimilaritiesPerRow", String.valueOf(maxSimilarItemsPerItem),
        "--excludeSelfSimilarity", String.valueOf(Boolean.TRUE),
        "--threshold", String.valueOf(threshold),
        "--randomSeed", String.valueOf(randomSeed),
        "--tempDir", getTempPath().toString(),
      });
    }

    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Job mostSimilarItems = prepareJob(similarityMatrixPath, getOutputPath(), SequenceFileInputFormat.class,
          MostSimilarItemPairsMapper.class, EntityEntityWritable.class, DoubleWritable.class,
          MostSimilarItemPairsReducer.class, EntityEntityWritable.class, DoubleWritable.class, TextOutputFormat.class);
      Configuration mostSimilarItemsConf = mostSimilarItems.getConfiguration();
      mostSimilarItemsConf.set(ITEM_ID_INDEX_PATH_STR,
          new Path(prepPath, PreparePreferenceMatrixJob.ITEMID_INDEX).toString());
      mostSimilarItemsConf.setInt(MAX_SIMILARITIES_PER_ITEM, maxSimilarItemsPerItem);
      boolean succeeded = mostSimilarItems.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
    }

    return 0;
  }

  public static class MostSimilarItemPairsMapper
      extends Mapper<IntWritable,VectorWritable,EntityEntityWritable,DoubleWritable> {

    private OpenIntLongHashMap indexItemIDMap;
    private int maxSimilarItemsPerItem;

    @Override
    protected void setup(Context ctx) {
      Configuration conf = ctx.getConfiguration();
      maxSimilarItemsPerItem = conf.getInt(MAX_SIMILARITIES_PER_ITEM, -1);
      indexItemIDMap = TasteHadoopUtils.readIDIndexMap(conf.get(ITEM_ID_INDEX_PATH_STR), conf);

      Preconditions.checkArgument(maxSimilarItemsPerItem > 0, "maxSimilarItemsPerItem must be greater then 0!");
    }

    @Override
    protected void map(IntWritable itemIDIndexWritable, VectorWritable similarityVector, Context ctx)
      throws IOException, InterruptedException {

      int itemIDIndex = itemIDIndexWritable.get();

      TopSimilarItemsQueue topKMostSimilarItems = new TopSimilarItemsQueue(maxSimilarItemsPerItem);

      for (Vector.Element element : similarityVector.get().nonZeroes()) {
        SimilarItem top = topKMostSimilarItems.top();
        double candidateSimilarity = element.get();
        if (candidateSimilarity > top.getSimilarity()) {
          top.set(indexItemIDMap.get(element.index()), candidateSimilarity);
          topKMostSimilarItems.updateTop();
        }
      }

      long itemID = indexItemIDMap.get(itemIDIndex);
      for (SimilarItem similarItem : topKMostSimilarItems.getTopItems()) {
        long otherItemID = similarItem.getItemID();
        if (itemID < otherItemID) {
          ctx.write(new EntityEntityWritable(itemID, otherItemID), new DoubleWritable(similarItem.getSimilarity()));
        } else {
          ctx.write(new EntityEntityWritable(otherItemID, itemID), new DoubleWritable(similarItem.getSimilarity()));
        }
      }
    }
  }

  public static class MostSimilarItemPairsReducer
      extends Reducer<EntityEntityWritable,DoubleWritable,EntityEntityWritable,DoubleWritable> {
    @Override
    protected void reduce(EntityEntityWritable pair, Iterable<DoubleWritable> values, Context ctx)
      throws IOException, InterruptedException {
      ctx.write(pair, values.iterator().next());
    }
  }
}
