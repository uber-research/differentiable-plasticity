# Uber-only code for interacting with hdfs
#
# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import os.path

def checkHdfs():
    return os.path.isfile('/opt/hadoop/latest/bin/hdfs')

def transferFileToHdfsPath(sourcepath, targetpath):
    hdfspath = targetpath
    targetdir = os.path.dirname(targetpath)
    os.system('/opt/hadoop/latest/bin/hdfs dfs -mkdir -p {}'.format(targetdir))
    result = os.system(
        '/opt/hadoop/latest/bin/hdfs dfs -copyFromLocal -f {} {}'.format(sourcepath, hdfspath)
    )
    if result != 0:
        raise OSError('Cannot copyFromLocal {} {} returned {}'.format(sourcepath, hdfspath, result))

def transferFileToHdfsDir(sourcepath, targetdir):
    hdfspath = os.path.join(targetdir, os.path.basename(sourcepath))
    os.system('/opt/hadoop/latest/bin/hdfs dfs -mkdir -p {}'.format(targetdir))
    result = os.system(
        '/opt/hadoop/latest/bin/hdfs dfs -copyFromLocal -f {} {}'.format(sourcepath, hdfspath)
    )
    if result != 0:
        raise OSError('Cannot copyFromLocal {} {} returned {}'.format(sourcepath, hdfspath, result))

