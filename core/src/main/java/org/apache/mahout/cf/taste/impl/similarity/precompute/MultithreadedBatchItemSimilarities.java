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

package org.apache.mahout.cf.taste.impl.similarity.precompute;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.collect.Lists;
import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.precompute.BatchItemSimilarities;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItems;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItemsWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Precompute item similarities in parallel on a single machine. The recommender given to this class must use a
 * DataModel that holds the interactions in memory (such as
 * {@link org.apache.mahout.cf.taste.impl.model.GenericDataModel} or
 * {@link org.apache.mahout.cf.taste.impl.model.file.FileDataModel}) as fast random access to the data is required
 */
public class MultithreadedBatchItemSimilarities extends BatchItemSimilarities {

  private int batchSize;

  private static final int DEFAULT_BATCH_SIZE = 100;

  private static final Logger log = LoggerFactory.getLogger(MultithreadedBatchItemSimilarities.class);

  /**
   * @param recommender recommender to use
   * @param similarItemsPerItem number of similar items to compute per item
   */
  public MultithreadedBatchItemSimilarities(ItemBasedRecommender recommender, int similarItemsPerItem) {
    this(recommender, similarItemsPerItem, DEFAULT_BATCH_SIZE);
  }

  /**
   * @param recommender recommender to use
   * @param similarItemsPerItem number of similar items to compute per item
   * @param batchSize size of item batches sent to worker threads
   */
  public MultithreadedBatchItemSimilarities(ItemBasedRecommender recommender, int similarItemsPerItem, int batchSize) {
    super(recommender, similarItemsPerItem);
    this.batchSize = batchSize;
  }

  @Override
  public int computeItemSimilarities(int degreeOfParallelism, int maxDurationInHours, SimilarItemsWriter writer)
    throws IOException {

    ExecutorService executorService = Executors.newFixedThreadPool(degreeOfParallelism + 1);

    Output output = null;
    try {
      writer.open();

      DataModel dataModel = getRecommender().getDataModel();

      BlockingQueue<long[]> itemsIDsInBatches = queueItemIDsInBatches(dataModel, batchSize);
      BlockingQueue<List<SimilarItems>> results = new LinkedBlockingQueue<List<SimilarItems>>();

      AtomicInteger numActiveWorkers = new AtomicInteger(degreeOfParallelism);
      for (int n = 0; n < degreeOfParallelism; n++) {
        executorService.execute(new SimilarItemsWorker(n, itemsIDsInBatches, results, numActiveWorkers));
      }

      output = new Output(results, writer, numActiveWorkers);
      executorService.execute(output);

    } catch (Exception e) {
      throw new IOException(e);
    } finally {
      executorService.shutdown();
      try {
        boolean succeeded = executorService.awaitTermination(maxDurationInHours, TimeUnit.HOURS);
        if (!succeeded) {
          throw new RuntimeException("Unable to complete the computation in " + maxDurationInHours + " hours!");
        }
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
      Closeables.close(writer, false);
    }

    return output.getNumSimilaritiesProcessed();
  }

  private static BlockingQueue<long[]> queueItemIDsInBatches(DataModel dataModel, int batchSize) throws TasteException {

    LongPrimitiveIterator itemIDs = dataModel.getItemIDs();
    int numItems = dataModel.getNumItems();

    BlockingQueue<long[]> itemIDBatches = new LinkedBlockingQueue<long[]>((numItems / batchSize) + 1);

    long[] batch = new long[batchSize];
    int pos = 0;
    while (itemIDs.hasNext()) {
      if (pos == batchSize) {
        itemIDBatches.add(batch.clone());
        pos = 0;
      }
      batch[pos] = itemIDs.nextLong();
      pos++;
    }
    int nonQueuedItemIDs = batchSize - pos;
    if (nonQueuedItemIDs > 0) {
      long[] lastBatch = new long[nonQueuedItemIDs];
      System.arraycopy(batch, 0, lastBatch, 0, nonQueuedItemIDs);
      itemIDBatches.add(lastBatch);
    }

    log.info("Queued {} items in {} batches", numItems, itemIDBatches.size());

    return itemIDBatches;
  }


  private static class Output implements Runnable {

    private final BlockingQueue<List<SimilarItems>> results;
    private final SimilarItemsWriter writer;
    private final AtomicInteger numActiveWorkers;
    private int numSimilaritiesProcessed = 0;

    Output(BlockingQueue<List<SimilarItems>> results, SimilarItemsWriter writer, AtomicInteger numActiveWorkers) {
      this.results = results;
      this.writer = writer;
      this.numActiveWorkers = numActiveWorkers;
    }

    private int getNumSimilaritiesProcessed() {
      return numSimilaritiesProcessed;
    }

    @Override
    public void run() {
      while (numActiveWorkers.get() != 0) {
        try {
          List<SimilarItems> similarItemsOfABatch = results.poll(10, TimeUnit.MILLISECONDS);
          if (similarItemsOfABatch != null) {
            for (SimilarItems similarItems : similarItemsOfABatch) {
              writer.add(similarItems);
              numSimilaritiesProcessed += similarItems.numSimilarItems();
            }
          }
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
    }
  }

  private class SimilarItemsWorker implements Runnable {

    private final int number;
    private final BlockingQueue<long[]> itemIDBatches;
    private final BlockingQueue<List<SimilarItems>> results;
    private final AtomicInteger numActiveWorkers;

    SimilarItemsWorker(int number, BlockingQueue<long[]> itemIDBatches, BlockingQueue<List<SimilarItems>> results,
        AtomicInteger numActiveWorkers) {
      this.number = number;
      this.itemIDBatches = itemIDBatches;
      this.results = results;
      this.numActiveWorkers = numActiveWorkers;
    }

    @Override
    public void run() {

      int numBatchesProcessed = 0;
      while (!itemIDBatches.isEmpty()) {
        try {
          long[] itemIDBatch = itemIDBatches.take();

          List<SimilarItems> similarItemsOfBatch = Lists.newArrayListWithCapacity(itemIDBatch.length);
          for (long itemID : itemIDBatch) {
            List<RecommendedItem> similarItems = getRecommender().mostSimilarItems(itemID, getSimilarItemsPerItem());

            similarItemsOfBatch.add(new SimilarItems(itemID, similarItems));
          }

          results.offer(similarItemsOfBatch);

          if (++numBatchesProcessed % 5 == 0) {
            log.info("worker {} processed {} batches", number, numBatchesProcessed);
          }

        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
      log.info("worker {} processed {} batches. done.", number, numBatchesProcessed);
      numActiveWorkers.decrementAndGet();
    }
  }
}
