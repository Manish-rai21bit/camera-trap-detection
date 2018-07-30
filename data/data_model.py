
def fetchcolvalue(filename, searchkey, output_col_no):
    csvfile = open(os.path.join(Project_filepath, filename), 'rb')
    csvdata = csv.reader(csvfile, delimiter=',')
    first_row = next(csvdata)
    #filename = open(kids_agefile, 'r')
    ct = 0
    for row in csvdata:
        if searchkey in row[ct]:
            colVal = row[output_col_no]
            break
        else:
            colVal = ""
            
    csvfile.close()
    return colVal

# URL fetching
def fetchurl(filename, searchkey):
    csvfile = open(os.path.join(Project_filepath, filename), 'rb')
    csvdata = csv.reader(csvfile, delimiter=',')
    first_row = next(csvdata)
    csvdata2 = sorted(csvdata, key=lambda row: (0, 1)) 
    for row in csvdata2:
        if row[0] == searchkey:
            URL = 'https://snapshotserengeti.s3.msi.umn.edu/' + row[1]
            break
        else:
            URL = ''
    
    csvfile.close()
    return URL

def csvtodict(Project_filepath, bb_data, concensus_data, all_images_data, images):
    lst = []
    event_dict = {}
    csvfile = open(os.path.join(Project_filepath, 'Data/Schneider_Data/TestGoldStandardBoundBoxCoord.csv'), 'rb')
    csvdata = csv.reader(csvfile, delimiter=',')
    first_row = next(csvdata)
    for row in csvdata:
        if row[0][0:10] not in event_dict:
            event_dict[row[0][0:10]] = {'metadata' : {"SiteID": fetchcolvalue(concensus_data, row[0][0:10], 3), 
                                  "DateTime": fetchcolvalue(concensus_data, row[0][0:10], 2),  
                                  "Season": fetchcolvalue(all_images_data, row[0][0:10], 1)[0:2]},
                                    'images' : [{"Path" : os.path.join(Project_filepath, 'Data/Schneider_Data/', row[0]),
                                "URL" : fetchurl(all_images_data, row[0][0:10]),
                                "dim_x" : fetchcolvalue(bb_data, row[0], 1),
                                "dim_y" : fetchcolvalue(bb_data, row[0], 2),
                                "image_label" : "tbd",
                                'observations' : [{"bb_xmin" : fetchcolvalue(bb_data, row[0], 4),
                                      "bb_ymin" : fetchcolvalue(bb_data, row[0], 5),
                                      "bb_xmax" : fetchcolvalue(bb_data, row[0], 6),
                                      "bb_ymax" : fetchcolvalue(bb_data, row[0], 7), 
                                      "bb_label" : fetchcolvalue(bb_data, row[0], 3)
                                     }]
                               }]
                                    }
    return event_dict

def dicttojson(event_dict):
    return json.dumps(event_dict)

def jsontodict(event_json):
    return json.loads(event_json)


def create_tf_example(data):
    # TODO(user): Populate the following variables from your example.
    height = int(data['dim_y']) # Image height
    width = int(data['dim_x']) # Image width
    filename = str(data['Path']) # Filename of the image. Empty if image is not from file
    #encoded_image_data = None # Encoded image bytes
    image_format = b'jpg' # b'jpeg' or b'png'

    xmins = [float(data['observations'][0]['bb_xmin'])] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [float(data['observations'][0]['bb_xmax'])] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [float(data['observations'][0]['bb_ymin'])] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [float(data['observations'][0]['bb_ymax'])] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    #classes_text = data[''] # List of string class name of bounding box (1 per box)
    #classes = data[''] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      #'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      #'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      #'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    return tf_example


def encode_to_tfr_record(test_feature):
    writer = tf.python_io.TFRecordWriter('test_event.tfrecord')
    example = create_tf_example(test_feature)
    writer.write(example.SerializeToString())
    writer.close()