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

