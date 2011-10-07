#!/bin/bash
# 
# Performs the setup procedures for clustering the ASF mail archives
# described in Taming Text.
# 
# Required Command-line Parameters:
#
#   $1 - Path to this script's working directory, you will need about
#        22GB of free space to run this script.
#
#   $2 - Path to where the ASF Public Archive data is, untarred.
#        If you are running Hadoop and the files are in HDFS, then
#        this will need to be an HDFS path.   Default is $1/input
#   $3 - Path to where this script saves the SequenceFile output.
#        If you are running Hadoop and you want the sequence files
#        saved to your HDFS then you need to set this value to an 
#        HDFS path and make sure you set HADOOP_HOME so Mahout can
#        find Hadoop.  Default is $1/sequence-files
#
#
# Required Environment Variables:
#
#   MAHOUT_HOME   
#          Root directory of your Mahout distribution
#
#   HADOOP_HOME
#          Only needed if you want to send output to HDFS
#
# Example:
#   ./prep_asf_mail_archives.sh /mnt/asf-mail-archives /mnt/asf-archives/asf-mail-archives-7-18-2011 /mnt/asf-mail-archives/output
#
#   This will download the TAR files from S3, extract them, and then
#   run the Mahout org.apache.mahout.text.SequenceFilesFromMailArchives job
#   to create Hadoop SequenceFiles in /mnt/asf-mail-archives/output
#
#/**
# * Licensed to the Apache Software Foundation (ASF) under one or more
# * contributor license agreements.  See the NOTICE file distributed with
# * this work for additional information regarding copyright ownership.
# * The ASF licenses this file to You under the Apache License, Version 2.0
# * (the "License"); you may not use this file except in compliance with
# * the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

if [ "$MAHOUT_HOME" = "" ]; then
  echo "Error: MAHOUT_HOME is not set."
  exit 1
fi

if [ "$1" = "" ]; then
  echo "Error: Please pass the path to your prep directory, such as /mnt/asf-mail-archives.\n\n\tUsage: $0 workingDir inputPath outputPath\n"
  exit 1
fi

# Location where this script saves files
PREP_DIR=$1

if [ "$2" != "" ]; then
  SEQFILE_INPUT_DIR=$2
else
  SEQFILE_INPUT_DIR=$PREP_DIR/input
fi


# Change this to an HDFS path if you are running Hadoop
if [ "$3" != "" ]; then
  SEQFILE_OUTPUT_DIR=$3
else
  SEQFILE_OUTPUT_DIR=$PREP_DIR/sequence-files
fi

# If output sent to HDFS, clear MAHOUT_LOCAL and make sure HADOOP_HOME is set
if [[ "$SEQFILE_OUTPUT_DIR" = hdfs://* ]]; then
  export MAHOUT_LOCAL=
  if [ "$HADOOP_HOME" = "" ]; then
    echo "Error: HADOOP_HOME must be set if you want to send output to HDFS."
    exit 1
  fi
else
  export MAHOUT_LOCAL=$PREP_DIR  
fi

echo "Running $0 with:
  PREP_DIR = $PREP_DIR
  SEQFILE_INPUT_DIR = $SEQFILE_INPUT_DIR
  SEQFILE_OUTPUT_DIR = $SEQFILE_OUTPUT_DIR
  MAHOUT_LOCAL = $MAHOUT_LOCAL
  HADOOP_HOME = $HADOOP_HOME"

# Run Mahout in Local mode! Remove this if you want the
# sequence files stored in your HDFS


# convert the extracted gz files into Hadoop SequenceFiles
echo "Converting extracted directories to SequenceFiles ..."
$MAHOUT_HOME/bin/mahout org.apache.mahout.text.SequenceFilesFromMailArchives \
--input $SEQFILE_INPUT_DIR --output $SEQFILE_OUTPUT_DIR --subject --body \
-c UTF-8 -chunk 1024 -prefix asf_archives
