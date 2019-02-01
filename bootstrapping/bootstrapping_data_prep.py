"""module for identifying images to be shown to volunteers and images for the training loop"""
import csv


def pred_groundtruth_consolidate_csv_to_dict(pred_groundtruth_consolidate_csv):
    """This function takes the csv with groundtruth boxes and creates a dictionary object 
    with labels, groundtruth_count and prediction_count lists in matching orders.
    For Example: 
    {'groundtruth_counts': ['6', '', ''], 'prediction_counts': ['3', '1', '1'], 'labels': ['7', '2', '20']}
    We look at this output dictionary element like this:
    There are 3 labels in this image/filename - '7', '2', '20'
    For '7' there are 6 boxes in groundtruth, and 3 boxes in prediction
    For label '2' and '20', there are no groundtruth boxes but the model made false predictions (False positives)
    """
    with open(pred_groundtruth_consolidate_csv, 'r') as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',')
        header = next(csvdata)

        pred_groundtruth_consolidate_dict = {}

        for i, row in enumerate(csvdata):
            if row[0] not in pred_groundtruth_consolidate_dict:
                pred_groundtruth_consolidate_dict[row[0]] = {}
                pred_groundtruth_consolidate_dict[row[0]]['labels'] = []
                pred_groundtruth_consolidate_dict[row[0]]['groundtruth_counts'] = []
                pred_groundtruth_consolidate_dict[row[0]]['prediction_counts'] = []

            pred_groundtruth_consolidate_dict[row[0]]['labels'].append(row[2])
            pred_groundtruth_consolidate_dict[row[0]]['groundtruth_counts'].append(row[3])
            pred_groundtruth_consolidate_dict[row[0]]['prediction_counts'].append(row[4])

        return pred_groundtruth_consolidate_dict
            

def get_correct_incorrect_images(pred_groundtruth_consolidate_dict):
    """Return a list of images/filenames that have correct, incorrect, and corrected predictions.
    corrected_image_species_list: list of tuples with corrected filename and corrected species label"""
    correct_list, corrected_image_species_list, incorrect_list = [], [], []
    
    for filename, value in pred_groundtruth_consolidate_dict.items():
        use_for_train_flag = False
        # Checking out the perfect matches
        if value['prediction_counts'] == value['groundtruth_counts']:
            correct_list.append(filename)
        elif value['prediction_counts'] != value['groundtruth_counts'] \
                and len(list(filter(None, value['groundtruth_counts']))) == 1 \
                and '11-50' not in value['groundtruth_counts'] \
                and '51+' not in value['groundtruth_counts'] \
                and '11-50' not in value['prediction_counts'] \
                and '51+' not in value['prediction_counts']:
            # if sum of count of boxes in prediction and groundtruth are the same then this can be easily corrected
            if sum(pd.to_numeric(list(filter(None, value['groundtruth_counts'])))) == sum(pd.to_numeric(list(filter(None, value['prediction_counts'])))): 
                # This can be corrected and hence add the filename to the corrected list
                correct_label_index = next(i for i, v in enumerate(value['groundtruth_counts']) if v) # index of correct label
                corrected_image_species_list.append((filename, value['labels'][correct_label_index]))
            else:
                incorrect_list.append(filename)
        else:
            incorrect_list.append(filename)
            
    return correct_list, corrected_image_species_list, incorrect_list

def training_data_prep_from_correct_prediction(prediction_csv_path,
                                                correct_list, 
                                                corrected_image_species_list, 
                                                label_map
                                              ):
    """Takes in the original CSV with bounding box predictions, filters out the correct pridictions
    and corrects the labels for the images that are in the corrected_image_species_list.
    Returns a dataframe with correctly predicted bounding boxes that can be used to build the TFRecords
    
    prediction_csv_path '/home/ubuntu/data/tensorflow/my_workspace/training_demo/Predictions/snapshot_serengeti_test2.csv'
    """
    
    predicted_df = pd.read_csv(prediction_csv_path)
    inverse_label_map = {v: k for k, v in label_map.items()}

    corrected_image_list = [file[0] for file in corrected_image_species_list]
    
    correct_predicted_df = predicted_df[predicted_df['filename'].isin(correct_list + corrected_image_list)]
    correct_predicted_df_dict = correct_predicted_df.to_dict(orient='index')

    for i, val in correct_predicted_df_dict.items():
        correct_predicted_df_dict[i]['class'] = inverse_label_map[correct_predicted_df_dict[i]['labels']]
        if val['filename'] in corrected_image_list:
            correct_predicted_df_dict[i]['labels'] = [rec[1] for rec in corrected_image_species_list if val['filename'] in rec[0]][0]

    correct_predicted_final_df = pd.DataFrame.from_dict(correct_predicted_df_dict, orient='index')
    correct_predicted_final_df = correct_predicted_final_df[['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]
    
    return correct_predicted_final_df