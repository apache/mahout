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
#   $2 - Path to where this script saves the SequenceFile output.
#        If you are running Hadoop and you want the sequence files
#        saved to your HDFS then you need to set this value to an 
#        HDFS path and make sure you set HADOOP_HOME so Mahout can
#        find Hadoop.
#
#   In addition, you will need to install, configure and add s3cmd
#   to your PATH before running this script. s3cmd is needed to
#   download the TAR files from Amazon S3, for more information, see:
#      http://s3tools.org/s3cmd
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
#   ./prep_asf_mail_archives.sh /mnt/asf-mail-archives /mnt/asf-mail-archives/output
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

# Make sure they have s3cmd installed
command -v s3cmd >/dev/null || { 
  echo "Error: s3cmd command not found. See http://s3tools.org/s3cmd for more information."; 
  exit 1; 
}

if [ "$1" = "" ]; then
  echo "Error: Please pass the path to your prep directory, such as /mnt/asf-mail-archives.\n\n\tUsage: $0 workingDir outputPath\n"
  exit 1
fi

# Location where this script saves files
PREP_DIR=$1

# Change this to an HDFS path if you are running Hadoop
if [ "$2" != "" ]; then
  SEQFILE_OUTPUT_DIR=$2
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
  SEQFILE_OUTPUT_DIR = $SEQFILE_OUTPUT_DIR
  MAHOUT_LOCAL = $MAHOUT_LOCAL
  HADOOP_HOME = $HADOOP_HOME"

# Run Mahout in Local mode! Remove this if you want the
# sequence files stored in your HDFS

mkdir -p $PREP_DIR/downloads $PREP_DIR/extracted

# download the tar files from S3

cd $PREP_DIR/downloads
if [ ! -e public_a_d.tar ]
then
  echo "Downloading public_a_d.tar files from S3 to $PREP_DIR/downloads"
  s3cmd get s3://asf-mail-archives/public_a_d.tar || {
    echo "Download from S3 failed, check console for errors.";
    exit 1;
  }
fi
if [ ! -e public_e_k.tar ]
then
  echo "Downloading public_e_k.tar files from S3 to $PREP_DIR/downloads"
  s3cmd get s3://asf-mail-archives/public_e_k.tar || {
   echo "Download from S3 failed, check console for errors.";
   exit 1;
  }
fi
if [ ! -e public_l_o.tar ]
then
  echo "Downloading public_l_o.tar files from S3 to $PREP_DIR/downloads"
  s3cmd get s3://asf-mail-archives/public_l_o.tar || {
   echo "Download from S3 failed, check console for errors.";
    exit 1;
  }
fi
if [ ! -e public_s_t.tar ]
then
  echo "Downloading public_s_t.tar files from S3 to $PREP_DIR/downloads"
  s3cmd get s3://asf-mail-archives/public_s_t.tar || {
   echo "Download from S3 failed, check console for errors.";
    exit 1;
  }
fi
if [ ! -e public_u_z.tar ]
then
  echo "Downloading public_u_z.tar files from S3 to $PREP_DIR/downloads"
  s3cmd get s3://asf-mail-archives/public_u_z.tar || {
   echo "Download from S3 failed, check console for errors.";
    exit 1;
  }
fi



# extract the tar files to your local drive

cd $PREP_DIR/extracted
#check to see if we have already extracted
if [ ! -e "$PREP_DIR/extracted/abdera.apache.org" ]
then
  echo "Extracting tar files from $PREP_DIR/downloads/public_a_d.tar"
  tar xf $PREP_DIR/downloads/public_a_d.tar || {
    echo "Extract TAR files failed, check console for errors.";
    exit 1;
  }
fi
if [ ! -e "$PREP_DIR/extracted/excalibur.apache.org" ]
then
  echo "Extracting tar files from $PREP_DIR/downloads/public_e_k.tar"
  tar xf $PREP_DIR/downloads/public_e_k.tar || {
    echo "Extract TAR files failed, check console for errors.";
    exit 1;
  }
fi
if [ ! -e "$PREP_DIR/extracted/labs.apache.org" ]
then
  echo "Extracting tar files from $PREP_DIR/downloads/public_l_o.tar"
  tar xf $PREP_DIR/downloads/public_l_o.tar || {
    echo "Extract TAR files failed, check console for errors.";
    exit 1;
  }
fi
if [ ! -e "$PREP_DIR/extracted/shale.apache.org" ]
then
  echo "Extracting tar files from $PREP_DIR/downloads/public_s_t.tar"
  tar xf $PREP_DIR/downloads/public_s_t.tar || {
    echo "Extract TAR files failed, check console for errors.";
    exit 1;
  }
fi
if [ ! -e "$PREP_DIR/extracted/uima.apache.org" ]
then
  echo "Extracting tar files from $PREP_DIR/downloads/public_u_z.tar"
  tar xf $PREP_DIR/downloads/public_u_z.tar || {
    echo "Extract TAR files failed, check console for errors.";
    exit 1;
  }
fi

# convert the extracted gz files into Hadoop SequenceFiles
echo "Converting extracted directories to SequenceFiles ..."
$MAHOUT_HOME/bin/mahout org.apache.mahout.text.SequenceFilesFromMailArchives \
--input $PREP_DIR/extracted --output $SEQFILE_OUTPUT_DIR \
-c UTF-8 -chunk 1024 -prefix asf_archives
