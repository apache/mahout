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

package org.apache.mahout.cf.taste.example.email;

import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.VarIntWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Convert the Mail archives (see {@link org.apache.mahout.text.SequenceFilesFromMailArchives}) to a preference
 * file that can be consumed by the {@link org.apache.mahout.cf.taste.hadoop.pseudo.RecommenderJob}.
 * <p/>
 * This assumes the input is a Sequence File, that the key is: filename/message id and the value is a list
 * (separated by the user's choosing) containing the from email and any references
 * <p/>
 * The output is a matrix where either the from or to are the rows (represented as longs) and the columns are the
 * message ids that the user has interacted with (as a VectorWritable).  This class currently does not account for
 * thread hijacking.
 * <p/>
 * It also outputs a side table mapping the row ids to their original and the message ids to the message thread id
 */
public final class MailToPrefsDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(MailToPrefsDriver.class);

  private static final String OUTPUT_FILES_PATTERN = "part-*";
  private static final int DICTIONARY_BYTE_OVERHEAD = 4;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new MailToPrefsDriver(), args);
  }

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption("chunkSize", "cs", "The size of chunks to write.  Default is 100 mb", "100");
    addOption("separator", "sep", "The separator used in the input file to separate to, from, subject.  Default is \\n",
        "\n");
    addOption("from", "f", "The position in the input text (value) where the from email is located, starting from "
        + "zero (0).", "0");
    addOption("refs", "r", "The position in the input text (value) where the reference ids are located, "
        + "starting from zero (0).", "1");
    addOption(buildOption("useCounts", "u", "If set, then use the number of times the user has interacted with a "
        + "thread as an indication of their preference.  Otherwise, use boolean preferences.", false, false,
        String.valueOf(true)));
    Map<String, List<String>> parsedArgs = parseArguments(args);

    Path input = getInputPath();
    Path output = getOutputPath();
    int chunkSize = Integer.parseInt(getOption("chunkSize"));
    String separator = getOption("separator");
    Configuration conf = getConf();
    boolean useCounts = hasOption("useCounts");
    AtomicInteger currentPhase = new AtomicInteger();
    int[] msgDim = new int[1];
    //TODO: mod this to not do so many passes over the data.  Dictionary creation could probably be a chain mapper
    List<Path> msgIdChunks = null;
    boolean overwrite = hasOption(DefaultOptionCreator.OVERWRITE_OPTION);
    // create the dictionary between message ids and longs
    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      //TODO: there seems to be a pattern emerging for dictionary creation
      // -- sparse vectors from seq files also has this.
      Path msgIdsPath = new Path(output, "msgIds");
      if (overwrite) {
        HadoopUtil.delete(conf, msgIdsPath);
      }
      log.info("Creating Msg Id Dictionary");
      Job createMsgIdDictionary = prepareJob(input,
              msgIdsPath,
              SequenceFileInputFormat.class,
              MsgIdToDictionaryMapper.class,
              Text.class,
              VarIntWritable.class,
              MailToDictionaryReducer.class,
              Text.class,
              VarIntWritable.class,
              SequenceFileOutputFormat.class);

      boolean succeeded = createMsgIdDictionary.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
      //write out the dictionary at the top level
      msgIdChunks = createDictionaryChunks(msgIdsPath, output, "msgIds-dictionary-",
          createMsgIdDictionary.getConfiguration(), chunkSize, msgDim);
    }
    //create the dictionary between from email addresses and longs
    List<Path> fromChunks = null;
    if (shouldRunNextPhase(parsedArgs, currentPhase)) {
      Path fromIdsPath = new Path(output, "fromIds");
      if (overwrite) {
        HadoopUtil.delete(conf, fromIdsPath);
      }
      log.info("Creating From Id Dictionary");
      Job createFromIdDictionary = prepareJob(input,
              fromIdsPath,
              SequenceFileInputFormat.class,
              FromEmailToDictionaryMapper.class,
              Text.class,
              VarIntWritable.class,
              MailToDictionaryReducer.class,
              Text.class,
              VarIntWritable.class,
              SequenceFileOutputFormat.class);
      createFromIdDictionary.getConfiguration().set(EmailUtility.SEPARATOR, separator);
      boolean succeeded = createFromIdDictionary.waitForCompletion(true);
      if (!succeeded) {
        return -1;
      }
      //write out the dictionary at the top level
      int[] fromDim = new int[1];
      fromChunks = createDictionaryChunks(fromIdsPath, output, "fromIds-dictionary-",
          createFromIdDictionary.getConfiguration(), chunkSize, fromDim);
    }
    //OK, we have our dictionaries, let's output the real thing we need: <from_id -> <msgId, msgId, msgId, ...>>
    if (shouldRunNextPhase(parsedArgs, currentPhase) && fromChunks != null && msgIdChunks != null) {
      //Job map
      //may be a way to do this so that we can load the from ids in memory, if they are small enough so that
      // we don't need the double loop
      log.info("Creating recommendation matrix");
      Path vecPath = new Path(output, "recInput");
      if (overwrite) {
        HadoopUtil.delete(conf, vecPath);
      }
      //conf.set(EmailUtility.FROM_DIMENSION, String.valueOf(fromDim[0]));
      conf.set(EmailUtility.MSG_ID_DIMENSION, String.valueOf(msgDim[0]));
      conf.set(EmailUtility.FROM_PREFIX, "fromIds-dictionary-");
      conf.set(EmailUtility.MSG_IDS_PREFIX, "msgIds-dictionary-");
      conf.set(EmailUtility.FROM_INDEX, getOption("from"));
      conf.set(EmailUtility.REFS_INDEX, getOption("refs"));
      conf.set(EmailUtility.SEPARATOR, separator);
      conf.set(MailToRecReducer.USE_COUNTS_PREFERENCE, String.valueOf(useCounts));
      int j = 0;
      int i = 0;
      for (Path fromChunk : fromChunks) {
        for (Path idChunk : msgIdChunks) {
          Path out = new Path(vecPath, "tmp-" + i + '-' + j);
          DistributedCache.setCacheFiles(new URI[]{fromChunk.toUri(), idChunk.toUri()}, conf);
          Job createRecMatrix = prepareJob(input, out, SequenceFileInputFormat.class,
                  MailToRecMapper.class, Text.class, LongWritable.class, MailToRecReducer.class, Text.class,
                  NullWritable.class, TextOutputFormat.class);
          createRecMatrix.getConfiguration().set("mapred.output.compress", "false");
          boolean succeeded = createRecMatrix.waitForCompletion(true);
          if (!succeeded) {
            return -1;
          }
          //copy the results up a level
          //HadoopUtil.copyMergeSeqFiles(out.getFileSystem(conf), out, vecPath.getFileSystem(conf), outPath, true,
          // conf, "");
          FileStatus[] fs = HadoopUtil.getFileStatus(new Path(out, "*"), PathType.GLOB, PathFilters.partFilter(), null,
              conf);
          for (int k = 0; k < fs.length; k++) {
            FileStatus f = fs[k];
            Path outPath = new Path(vecPath, "chunk-" + i + '-' + j + '-' + k);
            FileUtil.copy(f.getPath().getFileSystem(conf), f.getPath(), outPath.getFileSystem(conf), outPath, true,
                overwrite, conf);
          }
          HadoopUtil.delete(conf, out);
          j++;
        }
        i++;
      }
      //concat the files together
      /*Path mergePath = new Path(output, "vectors.dat");
      if (overwrite) {
        HadoopUtil.delete(conf, mergePath);
      }
      log.info("Merging together output vectors to vectors.dat in {}", output);*/
      //HadoopUtil.copyMergeSeqFiles(vecPath.getFileSystem(conf), vecPath, mergePath.getFileSystem(conf), mergePath,
      // false, conf, "\n");
    }

    return 0;
  }

  private static List<Path> createDictionaryChunks(Path inputPath,
                                                   Path dictionaryPathBase,
                                                   String name,
                                                   Configuration baseConf,
                                                   int chunkSizeInMegabytes, int[] maxTermDimension)
    throws IOException {
    List<Path> chunkPaths = Lists.newArrayList();

    Configuration conf = new Configuration(baseConf);

    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    long chunkSizeLimit = chunkSizeInMegabytes * 1024L * 1024L;
    int chunkIndex = 0;
    Path chunkPath = new Path(dictionaryPathBase, name + chunkIndex);
    chunkPaths.add(chunkPath);

    SequenceFile.Writer dictWriter = new SequenceFile.Writer(fs, conf, chunkPath, Text.class, IntWritable.class);

    try {
      long currentChunkSize = 0;
      Path filesPattern = new Path(inputPath, OUTPUT_FILES_PATTERN);
      int i = 1; //start at 1, since a miss in the OpenObjectIntHashMap returns a 0
      for (Pair<Writable, Writable> record
              : new SequenceFileDirIterable<Writable, Writable>(filesPattern, PathType.GLOB, null, null, true, conf)) {
        if (currentChunkSize > chunkSizeLimit) {
          Closeables.close(dictWriter, false);
          chunkIndex++;

          chunkPath = new Path(dictionaryPathBase, name + chunkIndex);
          chunkPaths.add(chunkPath);

          dictWriter = new SequenceFile.Writer(fs, conf, chunkPath, Text.class, IntWritable.class);
          currentChunkSize = 0;
        }

        Writable key = record.getFirst();
        int fieldSize = DICTIONARY_BYTE_OVERHEAD + key.toString().length() * 2 + Integer.SIZE / 8;
        currentChunkSize += fieldSize;
        dictWriter.append(key, new IntWritable(i++));
      }
      maxTermDimension[0] = i;
    } finally {
      Closeables.close(dictWriter, false);
    }

    return chunkPaths;
  }

}
