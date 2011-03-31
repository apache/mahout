package org.apache.mahout.common.iterator.sequencefile;

import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.iterator.ArrayIterator;
import org.apache.mahout.common.iterator.DelegatingIterator;
import org.apache.mahout.common.iterator.IteratorsIterator;
import org.apache.mahout.common.iterator.TransformingIterator;

/**
 * Like {@link SequenceFileValueIterator}, but iterates not just over one sequence file, but many. The input path
 * may be specified as a directory of files to read, or as a glob pattern. The set of files may be optionally
 * restricted with a {@link PathFilter}.
 */
public final class SequenceFileDirValueIterator<V extends Writable> extends DelegatingIterator<V> {

  public SequenceFileDirValueIterator(Path path,
                                      PathType pathType,
                                      PathFilter filter,
                                      Comparator<FileStatus> ordering,
                                      boolean reuseKeyValueInstances,
                                      Configuration conf)
    throws IOException {
    super(SequenceFileDirValueIterator.<V>buildDelegate(path,
                                                        pathType,
                                                        filter,
                                                        ordering,
                                                        reuseKeyValueInstances,
                                                        conf));
  }

  private static <V extends Writable> Iterator<V> buildDelegate(
      Path path,
      PathType pathType,
      PathFilter filter,
      Comparator<FileStatus> ordering,
      boolean reuseKeyValueInstances,
      Configuration conf) throws IOException {

    FileSystem fs = path.getFileSystem(conf);
    path = path.makeQualified(fs);
    FileStatus[] statuses;
    if (filter == null) {
      statuses = pathType == PathType.GLOB ? fs.globStatus(path) : fs.listStatus(path);
    } else {
      statuses = pathType == PathType.GLOB ? fs.globStatus(path, filter) : fs.listStatus(path, filter);
    }
    if (ordering != null) {
      Arrays.sort(statuses, ordering);
    }
    Iterator<FileStatus> fileStatusIterator = new ArrayIterator<FileStatus>(statuses);
    return new IteratorsIterator<V>(
        new FileStatusToSFIterator<V>(fileStatusIterator, reuseKeyValueInstances, conf));
  }


  private static class FileStatusToSFIterator<V extends Writable>
    extends TransformingIterator<FileStatus,Iterator<V>> {

    private final Configuration conf;
    private final boolean reuseKeyValueInstances;

    private FileStatusToSFIterator(Iterator<FileStatus> fileStatusIterator,
                                   boolean reuseKeyValueInstances,
                                   Configuration conf) {
      super(fileStatusIterator);
      this.reuseKeyValueInstances = reuseKeyValueInstances;
      this.conf = conf;
    }

    @Override
    protected Iterator<V> transform(FileStatus in) {
      try {
        return new SequenceFileValueIterator<V>(in.getPath(), reuseKeyValueInstances, conf);
      } catch (IOException ioe) {
        throw new IllegalStateException(ioe);
      }
    }
  }

}
