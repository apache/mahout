



## `FacesToVectorsApp`

#### CLI Parameters

- `c` or `--cascadeFilterPath` **required**. Path to OpenCV Cascade Filter to use, e.g. $OPENCV_3_0/data/haarcascades/haarcascade_frontalface_alt.xml
- `e` or `--eigenfacesPath` **required**. Path to output of eigenfaces file, e.g. $CYLON_HOME/data/eigenfaces.mmat
- `i` or `--inputVideoURL` **optional**. URL of input video, use `http://bglive-a.bitgravity.com/ndtv/247hi/live/native` for testing, defaults to `rtsp://192.168.100.1:554/cam1/mpeg4` (drone cam address)
- `k` or `--kakfaKey` **optional**. Kakfa Key to Write To. Default `testKey`
- `m` or `--colCentersPath` **required**. Path to output of col centers file that was generated with eigenfaces file, e.g. `$CYLON_HOME/data/colMeans.mmat`
- `s` or `--solrURL` **required**. URL of Solr, e.g. `http://localhost:8983/solr/cylonfaces`
- `t` or `--kakfaTopic` **optional**. Kakfa Topic to Write To. Default `testTopic`

#### Effects

Writes original frames to Kafka

_Topic_: `${kafkaTopic}-raw_image`

_Key_: `{kafkaKey}-frame`

_Message_: `BufferedImage` serialized to `Array[Byte]`

