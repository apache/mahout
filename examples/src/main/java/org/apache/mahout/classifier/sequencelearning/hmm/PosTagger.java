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

package org.apache.mahout.classifier.sequencelearning.hmm;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.io.CharStreams;
import com.google.common.io.Resources;
import org.apache.mahout.math.Matrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class implements a sample program that uses a pre-tagged training data
 * set to train an HMM model as a POS tagger. The training data is automatically
 * downloaded from the following URL:
 * http://www.jaist.ac.jp/~hieuxuan/flexcrfs/CoNLL2000-NP/train.txt It then
 * trains an HMM Model using supervised learning and tests the model on the
 * following test data set:
 * http://www.jaist.ac.jp/~hieuxuan/flexcrfs/CoNLL2000-NP/test.txt Further
 * details regarding the data files can be found at
 * http://flexcrfs.sourceforge.net/#Case_Study
 */
public final class PosTagger {

  private static final Logger log = LoggerFactory.getLogger(PosTagger.class);

  private static final Pattern SPACE = Pattern.compile(" ");
  private static final Pattern SPACES = Pattern.compile("[ ]+");

  /**
   * No public constructors for utility classes.
   */
  private PosTagger() {
    // nothing to do here really.
  }

  /**
   * Model trained in the example.
   */
  private static HmmModel taggingModel;

  /**
   * Map for storing the IDs for the POS tags (hidden states)
   */
  private static Map<String, Integer> tagIDs;

  /**
   * Counter for the next assigned POS tag ID The value of 0 is reserved for
   * "unknown POS tag"
   */
  private static int nextTagId;

  /**
   * Map for storing the IDs for observed words (observed states)
   */
  private static Map<String, Integer> wordIDs;

  /**
   * Counter for the next assigned word ID The value of 0 is reserved for
   * "unknown word"
   */
  private static int nextWordId = 1; // 0 is reserved for "unknown word"

  /**
   * Used for storing a list of POS tags of read sentences.
   */
  private static List<int[]> hiddenSequences;

  /**
   * Used for storing a list of word tags of read sentences.
   */
  private static List<int[]> observedSequences;

  /**
   * number of read lines
   */
  private static int readLines;

  /**
   * Given an URL, this function fetches the data file, parses it, assigns POS
   * Tag/word IDs and fills the hiddenSequences/observedSequences lists with
   * data from those files. The data is expected to be in the following format
   * (one word per line): word pos-tag np-tag sentences are closed with the .
   * pos tag
   *
   * @param url       Where the data file is stored
   * @param assignIDs Should IDs for unknown words/tags be assigned? (Needed for
   *                  training data, not needed for test data)
   * @throws IOException in case data file cannot be read.
   */
  private static void readFromURL(String url, boolean assignIDs) throws IOException {
    // initialize the data structure
    hiddenSequences = Lists.newLinkedList();
    observedSequences = Lists.newLinkedList();
    readLines = 0;

    // now read line by line of the input file
    List<Integer> observedSequence = Lists.newLinkedList();
    List<Integer> hiddenSequence = Lists.newLinkedList();

    for (String line : CharStreams.readLines(Resources.newReaderSupplier(new URL(url), Charsets.UTF_8))) {
      if (line.isEmpty()) {
        // new sentence starts
        int[] observedSequenceArray = new int[observedSequence.size()];
        int[] hiddenSequenceArray = new int[hiddenSequence.size()];
        for (int i = 0; i < observedSequence.size(); ++i) {
          observedSequenceArray[i] = observedSequence.get(i);
          hiddenSequenceArray[i] = hiddenSequence.get(i);
        }
        // now register those arrays
        hiddenSequences.add(hiddenSequenceArray);
        observedSequences.add(observedSequenceArray);
        // and reset the linked lists
        observedSequence.clear();
        hiddenSequence.clear();
        continue;
      }
      readLines++;
      // we expect the format [word] [POS tag] [NP tag]
      String[] tags = SPACE.split(line);
      // when analyzing the training set, assign IDs
      if (assignIDs) {
        if (!wordIDs.containsKey(tags[0])) {
          wordIDs.put(tags[0], nextWordId++);
        }
        if (!tagIDs.containsKey(tags[1])) {
          tagIDs.put(tags[1], nextTagId++);
        }
      }
      // determine the IDs
      Integer wordID = wordIDs.get(tags[0]);
      Integer tagID = tagIDs.get(tags[1]);
      // now construct the current sequence
      if (wordID == null) {
        observedSequence.add(0);
      } else {
        observedSequence.add(wordID);
      }

      if (tagID == null) {
        hiddenSequence.add(0);
      } else {
        hiddenSequence.add(tagID);
      }
    }

    // if there is still something in the pipe, register it
    if (!observedSequence.isEmpty()) {
      int[] observedSequenceArray = new int[observedSequence.size()];
      int[] hiddenSequenceArray = new int[hiddenSequence.size()];
      for (int i = 0; i < observedSequence.size(); ++i) {
        observedSequenceArray[i] = observedSequence.get(i);
        hiddenSequenceArray[i] = hiddenSequence.get(i);
      }
      // now register those arrays
      hiddenSequences.add(hiddenSequenceArray);
      observedSequences.add(observedSequenceArray);
    }
  }

  private static void trainModel(String trainingURL) throws IOException {
    tagIDs = Maps.newHashMapWithExpectedSize(44); // we expect 44 distinct tags
    wordIDs = Maps.newHashMapWithExpectedSize(19122); // we expect 19122
    // distinct words
    log.info("Reading and parsing training data file from URL: {}", trainingURL);
    long start = System.currentTimeMillis();
    readFromURL(trainingURL, true);
    long end = System.currentTimeMillis();
    double duration = (end - start) / 1000.0;
    log.info("Parsing done in {} seconds!", duration);
    log.info("Read {} lines containing {} sentences with a total of {} distinct words and {} distinct POS tags.",
             readLines, hiddenSequences.size(), nextWordId - 1, nextTagId - 1);
    start = System.currentTimeMillis();
    taggingModel = HmmTrainer.trainSupervisedSequence(nextTagId, nextWordId,
        hiddenSequences, observedSequences, 0.05);
    // we have to adjust the model a bit,
    // since we assume a higher probability that a given unknown word is NNP
    // than anything else
    Matrix emissions = taggingModel.getEmissionMatrix();
    for (int i = 0; i < taggingModel.getNrOfHiddenStates(); ++i) {
      emissions.setQuick(i, 0, 0.1 / taggingModel.getNrOfHiddenStates());
    }
    int nnptag = tagIDs.get("NNP");
    emissions.setQuick(nnptag, 0, 1 / (double) taggingModel.getNrOfHiddenStates());
    // re-normalize the emission probabilities
    HmmUtils.normalizeModel(taggingModel);
    // now register the names
    taggingModel.registerHiddenStateNames(tagIDs);
    taggingModel.registerOutputStateNames(wordIDs);
    end = System.currentTimeMillis();
    duration = (end - start) / 1000.0;
    log.info("Trained HMM models in {} seconds!", duration);
  }

  private static void testModel(String testingURL) throws IOException {
    log.info("Reading and parsing test data file from URL: {}", testingURL);
    long start = System.currentTimeMillis();
    readFromURL(testingURL, false);
    long end = System.currentTimeMillis();
    double duration = (end - start) / 1000.0;
    log.info("Parsing done in {} seconds!", duration);
    log.info("Read {} lines containing {} sentences.", readLines, hiddenSequences.size());

    start = System.currentTimeMillis();
    int errorCount = 0;
    int totalCount = 0;
    for (int i = 0; i < observedSequences.size(); ++i) {
      // fetch the viterbi path as the POS tag for this observed sequence
      int[] posEstimate = HmmEvaluator.decode(taggingModel, observedSequences.get(i), false);
      // compare with the expected
      int[] posExpected = hiddenSequences.get(i);
      for (int j = 0; j < posExpected.length; ++j) {
        totalCount++;
        if (posEstimate[j] != posExpected[j]) {
          errorCount++;
        }
      }
    }
    end = System.currentTimeMillis();
    duration = (end - start) / 1000.0;
    log.info("POS tagged test file in {} seconds!", duration);
    double errorRate = (double) errorCount / totalCount;
    log.info("Tagged the test file with an error rate of: {}", errorRate);
  }

  private static List<String> tagSentence(String sentence) {
    // first, we need to isolate all punctuation characters, so that they
    // can be recognized
    sentence = sentence.replaceAll("[,.!?:;\"]", " $0 ");
    sentence = sentence.replaceAll("''", " '' ");
    // now we tokenize the sentence
    String[] tokens = SPACES.split(sentence);
    // now generate the observed sequence
    int[] observedSequence = HmmUtils.encodeStateSequence(taggingModel, Arrays.asList(tokens), true, 0);
    // POS tag this observedSequence
    int[] hiddenSequence = HmmEvaluator.decode(taggingModel, observedSequence, false);
    // and now decode the tag names
    return HmmUtils.decodeStateSequence(taggingModel, hiddenSequence, false, null);
  }

  public static void main(String[] args) throws IOException {
    // generate the model from URL
    trainModel("http://www.jaist.ac.jp/~hieuxuan/flexcrfs/CoNLL2000-NP/train.txt");
    testModel("http://www.jaist.ac.jp/~hieuxuan/flexcrfs/CoNLL2000-NP/test.txt");
    // tag an exemplary sentence
    String test = "McDonalds is a huge company with many employees .";
    String[] testWords = SPACE.split(test);
    List<String> posTags = tagSentence(test);
    for (int i = 0; i < posTags.size(); ++i) {
      log.info("{}[{}]", testWords[i], posTags.get(i));
    }
  }

}
