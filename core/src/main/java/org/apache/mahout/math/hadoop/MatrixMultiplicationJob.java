package org.apache.mahout.math.hadoop;

import org.apache.commons.cli2.Option;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.join.CompositeInputFormat;
import org.apache.hadoop.mapred.join.TupleWritable;
import org.apache.hadoop.mapred.lib.MultipleInputs;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.AbstractJob;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;

public class MatrixMultiplicationJob extends AbstractJob {

  private static final String OUT_CARD = "output.vector.cardinality";

  private Map<String,String> argMap;

  public static JobConf createMatrixMultiplyJobConf(Path aPath, Path bPath, Path outPath, int outCardinality) {
    JobConf conf = new JobConf(MatrixMultiplicationJob.class);
    conf.setInputFormat(CompositeInputFormat.class);
    conf.set("mapred.join.expr", CompositeInputFormat.compose(
          "inner", SequenceFileInputFormat.class, new Path[] {aPath, bPath}));
    conf.setInt(OUT_CARD, outCardinality);
    conf.setOutputFormat(SequenceFileOutputFormat.class);
    FileOutputFormat.setOutputPath(conf, outPath);
    conf.setMapperClass(MatrixMultiplyMapper.class);
    conf.setCombinerClass(MatrixMultiplicationReducer.class);
    conf.setReducerClass(MatrixMultiplicationReducer.class);
    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(VectorWritable.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(VectorWritable.class);
    return conf;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new MatrixMultiplicationJob(), args);
  }

  @Override
  public int run(String[] strings) throws Exception {
    Option numRowsAOpt = buildOption("numRowsA",
                                     "nra",
                                     "Number of rows of the first input matrix");
    Option numColsAOpt = buildOption("numColsA",
                                     "nca",
                                     "Number of columns of the first input matrix");
    Option numRowsBOpt = buildOption("numRowsB",
                                     "nrb",
                                     "Number of rows of the second input matrix");

    Option numColsBOpt = buildOption("numColsB",
                                     "ncb",
                                     "Number of columns of the second input matrix");
    Option inputPathA = buildOption("inputPathA",
                                    "ia",
                                    "Path to the first input matrix");
    Option inputPathB = buildOption("inputPathB",
                                    "ib",
                                    "Path to the second input matrix");

    argMap = parseArguments(strings,
                            numRowsAOpt,
                            numRowsBOpt,
                            numColsAOpt,
                            numColsBOpt,
                            inputPathA,
                            inputPathB);

    DistributedRowMatrix a = new DistributedRowMatrix(argMap.get("--inputPathA"),
                                                      argMap.get("--tempDir"),
                                                      Integer.parseInt(argMap.get("--numRowsA")),
                                                      Integer.parseInt(argMap.get("--numColsA")));
    DistributedRowMatrix b = new DistributedRowMatrix(argMap.get("--inputPathB"),
                                                      argMap.get("--tempDir"),
                                                      Integer.parseInt(argMap.get("--numRowsB")),
                                                      Integer.parseInt(argMap.get("--numColsB")));

    a.configure(new JobConf(getConf()));
    b.configure(new JobConf(getConf()));

    DistributedRowMatrix c = a.times(b);

    return 0;
  }

  public static class MatrixMultiplyMapper extends MapReduceBase
      implements Mapper<IntWritable,TupleWritable,IntWritable,VectorWritable> {

    private int outCardinality;
    private final IntWritable row = new IntWritable();
    private final VectorWritable outVector = new VectorWritable();

    public void configure(JobConf conf) {
      outCardinality = conf.getInt(OUT_CARD, Integer.MAX_VALUE);
    }

    @Override
    public void map(IntWritable index,
                    TupleWritable v,
                    OutputCollector<IntWritable,VectorWritable> out,
                    Reporter reporter) throws IOException {
      boolean firstIsOutFrag =  ((VectorWritable)v.get(0)).get().size() == outCardinality;
      Vector outFrag = firstIsOutFrag ? ((VectorWritable)v.get(0)).get() : ((VectorWritable)v.get(1)).get();
      Vector multiplier = firstIsOutFrag ? ((VectorWritable)v.get(1)).get() : ((VectorWritable)v.get(0)).get();

      Iterator<Vector.Element> it = multiplier.iterateNonZero();
      while(it.hasNext()) {
        Vector.Element e = it.next();
        row.set(e.index());
        outVector.set(outFrag.times(e.get()));
        out.collect(row, outVector);
      }
    }
  }

  public static class MatrixMultiplicationReducer extends MapReduceBase
      implements Reducer<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    @Override
    public void reduce(IntWritable rowNum,
                       Iterator<VectorWritable> it,
                       OutputCollector<IntWritable,VectorWritable> out,
                       Reporter reporter) throws IOException {
      Vector accumulator;
      Vector row;
      if(it.hasNext()) {
        accumulator = new RandomAccessSparseVector(it.next().get());
      } else {
        return;
      }
      while(it.hasNext()) {
        row = it.next().get();
        row.addTo(accumulator);
      }
      out.collect(rowNum, new VectorWritable(new SequentialAccessSparseVector(accumulator)));
    }
  }

}
