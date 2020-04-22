/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
package org.apache.mahout.classifier.df.tools;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IOUtils;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

/**
 * This code is taken from the Apache Hadoop project.
 * It is taken from the class org.apache.hadoop.fs.FileUtil,
 * which is located in the hadoop-common library version 2.10.0
 * It is the last version of the code before it was removed in version 3.0.0
 */
public class HadoopFileUtil {

    public static boolean copyMerge(FileSystem srcFS, Path srcDir,
                                    FileSystem dstFS, Path dstFile,
                                    boolean deleteSource,
                                    Configuration conf, String addString) throws IOException {
        dstFile = checkDest(srcDir.getName(), dstFS, dstFile, false);

        if (!srcFS.getFileStatus(srcDir).isDirectory())
            return false;

        OutputStream out = dstFS.create(dstFile);

        try {
            FileStatus contents[] = srcFS.listStatus(srcDir);
            Arrays.sort(contents);
            for (int i = 0; i < contents.length; i++) {
                if (contents[i].isFile()) {
                    InputStream in = srcFS.open(contents[i].getPath());
                    try {
                        IOUtils.copyBytes(in, out, conf, false);
                        if (addString!=null)
                            out.write(addString.getBytes("UTF-8"));

                    } finally {
                        in.close();
                    }
                }
            }
        } finally {
            out.close();
        }

        if (deleteSource) {
            return srcFS.delete(srcDir, true);
        } else {
            return true;
        }
    }

    private static Path checkDest(String srcName, FileSystem dstFS, Path dst,
                                  boolean overwrite) throws IOException {
        FileStatus sdst;
        try {
            sdst = dstFS.getFileStatus(dst);
        } catch (FileNotFoundException e) {
            sdst = null;
        }
        if (null != sdst) {
            if (sdst.isDirectory()) {
                if (null == srcName) {
                    throw new PathIsDirectoryException(dst.toString());
                }
                return checkDest(null, dstFS, new Path(dst, srcName), overwrite);
            } else if (!overwrite) {
                throw new PathIOException(dst.toString(),
                        "Target " + dst + " already exists");
            }
        }
        return dst;
    }

}
