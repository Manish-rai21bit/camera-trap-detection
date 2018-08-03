
"""This function takes the GSSS bounding box dataset published by in the paper
    and converts it into a dictionary object. The column names in the CSV has to maintained"""
def csvtodict(Project_filepath, bb_data, concensus_data, all_images_data, images):
    lst = []
    event_dict = {}
    csvfile = open(os.path.join(Project_filepath, bb_data), 'r')
    csvdata = csv.reader(csvfile, delimiter=',')
    first_row = next(csvdata)
    for row in csvdata:
        if row[0] in lst2 and row[0][0:10] not in event_dict: # the condition in lst2 is to pick only the images usd by schneider
            event_dict[row[0][0:10]] = {'metadata' : {"SiteID": consensus_data[row[0][0:10]][0]['SiteID'],
                                  "DateTime": consensus_data[row[0][0:10]][0]['DateTime'], 
                                  "Season": all_images['ASG000c7bt'][0]['URL_Info'][0:2]},
                                    'images' : [{"Path" : os.path.join(Project_filepath, row[0]),
                                "URL" : Project_filepath + all_images[row[0][0:10]][0]['URL_Info'],
                                "dim_x" : gold_standard_bb[row[0]][0]['width'],
                                "dim_y" : gold_standard_bb[row[0]][0]['height'],
                                "image_label" : "tbd", # This is the primary label in case we want to have some for the whole image
                                'observations' : [{'bb_ymin': v['ymin'], 
                                                   'bb_ymax': v['ymax'], 
                                                      'bb_primary_label': v['class'], 
                                                      'bb_xmin': v['xmin'], 
                                                      'bb_xmax': v['xmax'], 
                                                      'bb_label': {"species" : v['class'],
                                                    "pose" : "standing/ sitting/ running"
                                                }} for k, v in enumerate(gold_standard_bb[row[0]])]
                               }]
                                    }
    return event_dict

"""This function writes the dictionary object to a json file.
    The outful file will act as a centeralized repository for building training dataset."""
def dicttojson(event_dict):
    with open('event_dict.json', 'w') as outfile:
        json.dump(event_dict, outfile)
    #return json.dump(event_dict, outfile)

"""This function creates a dictionary from the json dump.
    The dictionary object can be fed into the input pipeline to
    create a TFRecord file."""
def jsontodict(event_json):
    with open(event_json, 'r') as f:
        return json.load(f)

""" This function creates a tfrecord example from the dictionary element!"""
def create_tf_example(data_dict):
    #with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
    encoded_jpg = resize_jpeg((data_dict['images'][0]['Path']),  1000)
    #encoded_jpg_io = io.BytesIO(encoded_jpg)
    #image = Image.open(encoded_jpg_io)
    #width, height = image.size
    width = int(data_dict['images'][0]['dim_x'])
    height = int(data_dict['images'][0]['dim_y'])

    filename = data_dict['images'][0]['Path'].encode('utf-8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for bb_record in data_dict['images'][0]['observations']:
        xmins.append(float(bb_record['bb_xmin']) / width)
        xmaxs.append(float(bb_record['bb_xmax']) / width)
        ymins.append(float(bb_record['bb_ymin']) / height)
        ymaxs.append(float(bb_record['bb_ymax']) / height)
        classes_text.append(bb_record['bb_primary_label'].encode('utf8'))
        classes.append(class_text_to_int(bb_record['bb_primary_label']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    
    return tf_example

"""This iterates over each dictionary item, creates tf examples, 
    serializes the tfrecord examples and writes to a tfrecord file!!!
    As of now, it saves the TFRecord file in the home directory where the code is executed"""
def encode_to_tfr_record(test_feature, out_tfr_file):
    with tf.python_io.TFRecordWriter(out_tfr_file) as writer:
        count = 0
        for k, v in test_feature.items():
            count+=1
            if count%500==0:
                print("processing event number %s : %s" % (count, k))
            example = create_tf_example(v)
            writer.write(example.SerializeToString())
            