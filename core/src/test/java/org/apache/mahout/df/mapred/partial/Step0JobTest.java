package org.apache.mahout.df.mapred.partial;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import junit.framework.TestCase;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.df.data.DataConverter;
import org.apache.mahout.df.data.DataLoader;
import org.apache.mahout.df.data.Dataset;
import org.apache.mahout.df.data.Utils;
import org.apache.mahout.df.mapred.Builder;
import org.apache.mahout.df.mapred.partial.Step0Job.Step0Mapper;
import org.apache.mahout.df.mapred.partial.Step0Job.Step0Output;

public class Step0JobTest extends TestCase {

  // the generated data must be big enough to be splited by FileInputFormat

  private static final int numAttributes = 40;

  private static final int numInstances = 200;

  //int numTrees = 10;

  private static final int numMaps = 5;

  public void testStep0Mapper() throws Exception {
    Random rng = RandomUtils.getRandom();

    // create a dataset large enough to be split up
    String descriptor = Utils.randomDescriptor(rng, numAttributes);
    double[][] source = Utils.randomDoubles(rng, descriptor, numInstances);
    String[] sData = Utils.double2String(source);

    // write the data to a file
    Path dataPath = Utils.writeDataToTestFile(sData);
    
    JobConf job = new JobConf();
    job.setNumMapTasks(numMaps);

    FileInputFormat.setInputPaths(job, dataPath);

    // retrieve the splits
    TextInputFormat input = (TextInputFormat) job.getInputFormat();
    InputSplit[] splits = input.getSplits(job, numMaps);

    InputSplit[] sorted = Arrays.copyOf(splits, splits.length);
    Builder.sortSplits(sorted);

    Step0OutputCollector collector = new Step0OutputCollector(numMaps);
    Reporter reporter = Reporter.NULL;

    for (int p = 0; p < numMaps; p++) {
      InputSplit split = sorted[p];
      RecordReader<LongWritable, Text> reader = input.getRecordReader(split, job, reporter);

      LongWritable key = reader.createKey();
      Text value = reader.createValue();

      Step0Mapper mapper = new Step0Mapper();
      mapper.configure(p);

      Long firstKey = null;
      int size = 0;

      while (reader.next(key, value)) {
        if (firstKey == null) {
          firstKey = key.get();
        }

        mapper.map(key, value, collector, reporter);

        size++;
      }

      mapper.close();

      // validate the mapper's output
      assertEquals(p, collector.keys[p]);
      assertEquals(firstKey.longValue(), collector.values[p].firstId);
      assertEquals(size, collector.values[p].size);
    }

  }

  public void testProcessOutput() throws Exception {
    Random rng = RandomUtils.getRandom();

    // create a dataset large enough to be split up
    String descriptor = Utils.randomDescriptor(rng, numAttributes);
    double[][] source = Utils.randomDoubles(rng, descriptor, numInstances);

    // each instance label is its index in the dataset
    int labelId = Utils.findLabel(descriptor);
    for (int index = 0; index < numInstances; index++) {
      source[index][labelId] = index;
    }

    String[] sData = Utils.double2String(source);

    // write the data to a file
    Path dataPath = Utils.writeDataToTestFile(sData);
    
    // prepare a data converter
    Dataset dataset = DataLoader.generateDataset(descriptor, sData);
    DataConverter converter = new DataConverter(dataset);
    
    JobConf job = new JobConf();
    job.setNumMapTasks(numMaps);
    FileInputFormat.setInputPaths(job, dataPath);

    // retrieve the splits
    TextInputFormat input = (TextInputFormat) job.getInputFormat();
    InputSplit[] splits = input.getSplits(job, numMaps);

    InputSplit[] sorted = Arrays.copyOf(splits, splits.length);
    Builder.sortSplits(sorted);

    Reporter reporter = Reporter.NULL;

    int[] keys = new int[numMaps];
    Step0Output[] values = new Step0Output[numMaps];
    
    int[] expectedIds = new int[numMaps];
    
    for (int p = 0; p < numMaps; p++) {
      InputSplit split = sorted[p];
      RecordReader<LongWritable, Text> reader = input.getRecordReader(split, job, reporter);

      LongWritable key = reader.createKey();
      Text value = reader.createValue();

      Long firstKey = null;
      int size = 0;
      
      while (reader.next(key, value)) {
        if (firstKey == null) {
          firstKey = key.get();
          expectedIds[p] = converter.convert(0, value.toString()).label;
        }

        size++;
      }
      
      keys[p] = p;
      values[p] = new Step0Output(firstKey, size);
    }

    Step0Output[] partitions = Step0Job.processOutput(keys, values);
    
    int[] actualIds = Step0Output.extractFirstIds(partitions);
    
    assertTrue("Expected: " + Arrays.toString(expectedIds) + " But was: "
        + Arrays.toString(actualIds), Arrays.equals(expectedIds, actualIds));
  }

  protected static class Step0OutputCollector implements
      OutputCollector<IntWritable, Step0Output> {

    public final int[] keys;

    public final Step0Output[] values;

    private int index = 0;

    protected Step0OutputCollector(int numMaps) {
      keys = new int[numMaps];
      values = new Step0Output[numMaps];
    }

    @Override
    public void collect(IntWritable key, Step0Output value) throws IOException {
      if (index == keys.length) {
        throw new IOException("Received more output than expected : " + index);
      }

      keys[index] = key.get();
      values[index] = value.clone();

      index++;
    }

    /**
     * Number of outputs collected
     * 
     * @return
     */
    public int nbOutputs() {
      return index;
    }
  }
}
