
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
                                'observations' : [{"bb_xmin" : fetchcolvalue(bb_data, row[0], 4),
                                      "bb_ymin" : fetchcolvalue(bb_data, row[0], 5),
                                      "bb_xmax" : fetchcolvalue(bb_data, row[0], 6),
                                      "bb_ymax" : fetchcolvalue(bb_data, row[0], 7)
                                     }]
                               }]
                                    }
    return event_dict

def dicttojson(event_dict):
    return json.dumps(event_dict)

def jsontodict(event_json):
    data = json.loads(event_json)
    for key, value in data.iteritems(): 
        print key, value