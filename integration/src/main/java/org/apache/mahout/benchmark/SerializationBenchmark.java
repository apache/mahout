package org.apache.mahout.benchmark;

import static org.apache.mahout.benchmark.VectorBenchmarks.DENSE_VECTOR;
import static org.apache.mahout.benchmark.VectorBenchmarks.RAND_SPARSE_VECTOR;
import static org.apache.mahout.benchmark.VectorBenchmarks.SEQ_SPARSE_VECTOR;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.TimingStatistics;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.VectorWritable;

import com.google.common.io.Closeables;

public class SerializationBenchmark {
  public static final String SERIALIZE = "Serialize";
  public static final String DESERIALIZE = "Deserialize";
  private final VectorBenchmarks mark;

  public SerializationBenchmark(VectorBenchmarks mark) {
    this.mark = mark;
  }

  public void benchmark() throws IOException {
    serializeBenchmark();
    deserializeBenchmark();
  }

  public void serializeBenchmark() throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path("/tmp/dense-vector"), IntWritable.class,
        VectorWritable.class);

    Writable one = new IntWritable(0);
    VectorWritable vec = new VectorWritable();
    TimingStatistics stats = new TimingStatistics();

    try {
      for (int i = 0; i < mark.loop; i++) {
        TimingStatistics.Call call = stats.newCall(mark.leadTimeUsec);
        vec.set(mark.vectors[0][mark.vIndex(i)]);
        writer.append(one, vec);
        if (call.end(mark.maxTimeUsec)) {
          break;
        }
      }
    } finally {
      Closeables.close(writer, true);
    }
    mark.printStats(stats, SERIALIZE, DENSE_VECTOR);

    writer = new SequenceFile.Writer(fs, conf, new Path("/tmp/randsparse-vector"), IntWritable.class,
        VectorWritable.class);
    stats = new TimingStatistics();
    try {
      for (int i = 0; i < mark.loop; i++) {
        TimingStatistics.Call call = stats.newCall(mark.leadTimeUsec);
        vec.set(mark.vectors[1][mark.vIndex(i)]);
        writer.append(one, vec);
        if (call.end(mark.maxTimeUsec)) {
          break;
        }
      }
    } finally {
      Closeables.close(writer, true);
    }
    mark.printStats(stats, SERIALIZE, RAND_SPARSE_VECTOR);

    writer = new SequenceFile.Writer(fs, conf, new Path("/tmp/seqsparse-vector"), IntWritable.class,
        VectorWritable.class);
    stats = new TimingStatistics();
    try {
      for (int i = 0; i < mark.loop; i++) {
        TimingStatistics.Call call = stats.newCall(mark.leadTimeUsec);
        vec.set(mark.vectors[2][mark.vIndex(i)]);
        writer.append(one, vec);
        if (call.end(mark.maxTimeUsec)) {
          break;
        }
      }
    } finally {
      Closeables.close(writer, true);
    }
    mark.printStats(stats, SERIALIZE, SEQ_SPARSE_VECTOR);

  }

  public void deserializeBenchmark() throws IOException {
    doDeserializeBenchmark(DENSE_VECTOR, "/tmp/dense-vector");
    doDeserializeBenchmark(RAND_SPARSE_VECTOR, "/tmp/randsparse-vector");
    doDeserializeBenchmark(SEQ_SPARSE_VECTOR, "/tmp/seqsparse-vector");
  }

  private void doDeserializeBenchmark(String name, String pathString) throws IOException {
    TimingStatistics stats = new TimingStatistics();
    TimingStatistics.Call call = stats.newCall(mark.leadTimeUsec);
    SequenceFileValueIterator<Writable> iterator = new SequenceFileValueIterator<Writable>(new Path(pathString), true,
        new Configuration());
    while (iterator.hasNext()) {
      iterator.next();
      call.end();
      call = stats.newCall(mark.leadTimeUsec);
    }
    iterator.close();
    mark.printStats(stats, DESERIALIZE, name);
  }

}
