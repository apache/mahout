package org.apache.mahout.cf.taste.impl.recommender.slim;

import java.util.Collection;
import java.util.Map.Entry;
import java.util.concurrent.Callable;

import org.apache.commons.lang.math.RandomUtils;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseColumnMatrix;
import org.apache.mahout.math.jet.random.Normal;

/**
 * Common implementation for a SLIM optimizer.
 * 
 * @author Mihai Pitu
 *
 */
public abstract class AbstractOptimizer implements Optimizer {

  private final RefreshHelper refreshHelper;
  private FastByIDMap<Long> IDitemMapping;
  private FastByIDMap<Integer> itemIDMapping;
  protected final DataModel dataModel;
  private double mean;
  private double stDev;
  
  protected SlimSolution slim;
  private Normal normal;
  

  protected AbstractOptimizer(DataModel dataModel, double mean, double stDev)
      throws TasteException {
    this.dataModel = dataModel;
    this.mean = mean;
    this.stDev = stDev;
    buildMappings();
    refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        buildMappings();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
  }

  protected SlimSolution createSlimSolution(SparseColumnMatrix itemWeights) {
    return new SlimSolution(itemIDMapping, IDitemMapping, itemWeights);
  }
  
  public SlimSolution getSlimSolution() {
    return slim;
  }
    
  protected void prepareTraining() throws TasteException {
    int numItems = dataModel.getNumItems();
    // float[][] itemWeights = new float[numItems][numItems];
    //
    // // Initialize a double matrix with normal distributed Gaussian noise
    // // with the diagonal equal to 0
    this.normal = new Normal(mean, stDev,
        org.apache.mahout.common.RandomUtils.getRandom());

    SparseColumnMatrix itemWeights = new SparseColumnMatrix(numItems + 1, numItems + 1);

    slim = createSlimSolution(itemWeights);
  }

  @Override
  public synchronized double getAndInitWeightPos(Matrix itemWeights, int row, int column) {
    if (row == column)
      return 0;

    double weight = itemWeights.getQuick(row, column);
    if (weight == 0) {
      // weight = TestRandom.nextDouble();
      weight = Math.abs(normal.nextDouble());
      itemWeights.setQuick(row, column, weight);
    }
    return weight;
  }

  @Override
  public double getAndInitWeight(Matrix itemWeights, int row, int column) {
    if (row == column)
      return 0;

    double weight = itemWeights.getQuick(row, column);
    if (weight == 0) {
      // weight = TestRandom.nextDouble();
      weight = normal.nextDouble();
      itemWeights.setQuick(row, column, weight);
    }
    return weight;
  }

  private void buildMappings() throws TasteException {
    int numItems = dataModel.getNumItems();
    LongPrimitiveIterator it = dataModel.getItemIDs();
    itemIDMapping = createIDMapping(numItems, it);
    IDitemMapping = new FastByIDMap<Long>(dataModel.getNumItems());
    for (Entry<Long, Integer> entry : itemIDMapping.entrySet()) {
      IDitemMapping.put(entry.getValue(), entry.getKey());
    }
  }

  public long IDIndex(int itemIndex) throws NoSuchItemException {
    Long itemID = IDitemMapping.get(itemIndex);
    if (itemID == null) {
      throw new NoSuchItemException(itemIndex);
    }
    return itemID;
  }

  protected Integer itemIndex(long itemID) {
    Integer itemIndex = itemIDMapping.get(itemID);
    if (itemIndex == null) {
      itemIndex = itemIDMapping.size();
      itemIDMapping.put(itemID, itemIndex);
    }
    return itemIndex;
  }

  protected long sampleUserID() throws TasteException {
    // LongPrimitiveIterator userIDs =
    // SamplingLongPrimitiveIterator.maybeWrapIterator(dataModel.getUserIDs(),
    // 1);

    LongPrimitiveIterator it = dataModel.getUserIDs();
    int skip;
    do
      // skip = TestRandom.nextInt(dataModel.getNumUsers() + 1);
      skip = RandomUtils.nextInt(dataModel.getNumUsers() + 1);
    while (skip == 0);

    it.skip(skip - 1);
    return it.next();
  }

  protected int samplePosItemIndex(PreferenceArray userItems) {
    //int index = TestRandom.nextInt(userItems.length());
    int index = RandomUtils.nextInt(userItems.length());
    return itemIndex(userItems.getItemID(index));
  }

  protected int sampleNegItemIndex(PreferenceArray userItems)
      throws TasteException {
    int itemIndex;
    long itemID;
    do {
      // itemIndex = TestRandom.nextInt(dataModel.getNumItems() + 1);
      itemIndex = RandomUtils.nextInt(dataModel.getNumItems() + 1);
      itemID = IDIndex(itemIndex);
    } while (userItems.hasPrefWithItemID(itemID));

    return itemIndex;
  }

  public static final long BIASID = Long.MAX_VALUE - 1;

  private static FastByIDMap<Integer> createIDMapping(int size,
      LongPrimitiveIterator idIterator) {
    FastByIDMap<Integer> mapping = new FastByIDMap<Integer>(size);
    int index = 0;
    mapping.put(BIASID, index++);
    while (idIterator.hasNext()) {
      mapping.put(idIterator.nextLong(), index++);
    }
    return mapping;
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }

}
