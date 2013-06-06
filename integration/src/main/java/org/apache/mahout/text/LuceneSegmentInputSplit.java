package org.apache.mahout.text;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentInfoPerCommit;
import org.apache.lucene.index.SegmentInfos;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.List;

/**
 * {@link InputSplit} implementation that represents a Lucene segment.
 */
public class LuceneSegmentInputSplit extends InputSplit implements Writable {

  private Path indexPath;
  private String segmentInfoName;
  private long length;

  public LuceneSegmentInputSplit() {
    // For deserialization
  }

  public LuceneSegmentInputSplit(Path indexPath, String segmentInfoName, long length) {
    this.indexPath = indexPath;
    this.segmentInfoName = segmentInfoName;
    this.length = length;
  }

  @Override
  public long getLength() throws IOException, InterruptedException {
    return length;
  }

  @Override
  public String[] getLocations() throws IOException, InterruptedException {
    return new String[]{};
  }

  public String getSegmentInfoName() {
    return segmentInfoName;
  }

  public Path getIndexPath() {
    return indexPath;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeUTF(indexPath.toString());
    out.writeUTF(segmentInfoName);
    out.writeLong(length);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.indexPath = new Path(in.readUTF());
    this.segmentInfoName = in.readUTF();
    this.length = in.readLong();
  }

  /**
   * Get the {@link SegmentInfo} of this {@link InputSplit} via the given {@link Configuration}
   *
   * @param configuration the configuration used to locate the index
   * @return the segment info or throws exception if not found
   * @throws IOException if an error occurs when accessing the directory
   */
  public SegmentInfoPerCommit getSegment(Configuration configuration) throws IOException {
    ReadOnlyFileSystemDirectory directory = new ReadOnlyFileSystemDirectory(FileSystem.get(configuration), indexPath, false, configuration);

    SegmentInfos segmentInfos = new SegmentInfos();
    segmentInfos.read(directory);

    for (SegmentInfoPerCommit segmentInfo : segmentInfos) {
      if (segmentInfo.info.name.equals(segmentInfoName)) {
        return segmentInfo;
      }
    }

    throw new IllegalArgumentException("No such segment: '" + segmentInfoName + "' in directory " + directory.toString());
  }
}