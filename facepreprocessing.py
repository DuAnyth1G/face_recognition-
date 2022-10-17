from concurrent.futures import process
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing import image

import pandas as pd
import numpy as np
import pickle
import cv2
import os

###  secondary functions

def work_dir_generator(path):
    for root, dirs, files in os.walk(path):
        if len(dirs) == 0:
            yield root + "\\"

# collect image from dir
def collect_image_from_dir(db_path):

    employees = []

    for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
        for file in f:
            if ('.jpg' in file.lower()) or ('.png' in file.lower()):
                if r[-1] not in '\\/':
                    r += "\\"
                exact_path = r + file
                employees.append(exact_path)
    
    size = len(employees)
    if size == 0:
        raise ValueError("There is no image in ", db_path," folder! Validate .jpg or .png files exist in this path.")

    return employees, size

def get_path_to_res_plk(test_db_path, model_name, align, padd):
    
    file_name = 'res_data_' + model_name
    if align:
        file_name += '_align'
    if padd:
        file_name += '_pad'
    file_name += '.pkl'
    
    return test_db_path + file_name

def get_file_name(path):
    return os.path.basename(path).split('.')[-2]

def toBGR(img_cv2):
    return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

def img_to_bytes(img):
    return cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[1].tobytes()

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


from deepface import DeepFace
from deepface.commons import functions
#from deepface.detectors import FaceDetector, RetinaFaceWrapper as Retina

from retinaface import RetinaFace
from retinaface.commons import preprocess, postprocess

###  functions for face processing

# open original representations of current model 
def load_original(db_path = '0_ORIGINAL_PNG_250_200\\0_ORIGINAL_PNG_250_200\\'
    , model_name = 'ArcFace'
    , align = False
    , padd = False):

    file_name = 'representations_' + model_name
    if align:
        file_name += '_align'
    if padd:
        file_name += '_pad'
    file_name += '.pkl'
    file_name = file_name.replace("-", "_").lower()

    with open(db_path + file_name, "rb") as f:
        paths, representations = zip(*pickle.load(f))
    
    df = pd.DataFrame(data = zip(representations), columns = ["representation"], index = map(os.path.abspath, paths))
    
    return df.copy()


def load_face_object(dir = '0_ORIGINAL_PNG_250_200\\0_ORIGINAL_PNG_250_200\\') -> pd.DataFrame:

    if dir[-1] not in '\\/':
       dir += "\\"
    
    file_name = 'representations_retinaface_new_with_insurance.pkl'
    with open(dir + file_name, "rb") as f:
        return pickle.load(f)

# create representation.pkl of original
def represent_original(db_path = '0_ORIGINAL_PNG_250_200\\0_ORIGINAL_PNG_250_200\\'
    , align = False
    , padd = False
    , model_name = 'ArcFace'
    , file = None
    , model = None
    , distance_metric = 'cosine'
    , enforce_detection = True
    , progressbar_out = None
    ):


    models = {}
    models[model_name] = model
    model_names = []; metric_names = []
    model_names.append(model_name)
    metric_names.append(distance_metric)

    employees, size = collect_image_from_dir(db_path)

    face_detector = None
    if type(file) == type(None):
        face_detector = RetinaFace.build_model()
    
    representations = []

    if progressbar_out == None:
        progressbar_out = range(0, size)

	#for employee in employees:
    for index in progressbar_out:
        employee = employees[index]

        instance = []
        instance.append(os.path.abspath(employee))

        for j in model_names:
            custom_model = models[j]

            shape = functions.find_input_shape(custom_model)
            
            detected_face, face_region, info = preprocess_face(employee
                , target_size = shape
                , face_detector = face_detector
                , enforce_detection = enforce_detection
                , threshold = 0.9
                , file = file
                , align = align
                , padd = padd
                )

            representation = custom_model.predict(detected_face)[0].tolist()

            instance.append(representation)

        representations.append(instance)

    file_name = 'representations_' + model_name
    if align:
        file_name += '_align'
    if padd:
        file_name += '_pad'
    file_name += '.pkl'
    file_name = file_name.replace("-", "_").lower()
    full_path = db_path + file_name 
    with open(full_path, "wb") as f:
        pickle.dump(representations, f)

# detect and resize face on image
def preprocess_face(img_path         # path
    
    , target_size=(224, 224)     # recognizer model 
    
    , face_detector = None       # detector model
    , detector_backend = 'retinaface'

    , enforce_detection = True   # detection settings
    , threshold = 0.3
    , file = None
    , insurance_file = None
    , find_all = False

    , align = False              # postprocess
    , padd = False
    ):

    info = ''
    check_threshold = False
    get_original_obj = False
    img_abspath = os.path.abspath(img_path)
    img = functions.load_image(img_abspath)
    base_img = img.copy()
    img_region = [0, 0, img.shape[0], img.shape[1]]

    #----------------------------------------------
    #people would like to skip detection and alignment if they already have pre-processed images
    if detector_backend == 'skip':
        return img, img_region, detector_backend
    #----------------------------------------------

    #detector stored in a global variable in FaceDetector object.
    #this call should be completed very fast because it will return found in memory
    #it will not build face detector model in each call (consider for loops)

    if type(file) == type(None):
        obj = RetinaFace.detect_faces(img, 
            model = face_detector,
            threshold = threshold
        )
        info = 'from model'
    else:
        obj = file.loc[
            img_abspath,
            'retinaface_representation'
        ]
        info = 'from file'

    sorter = lambda item: item[1]['score']

    if type(obj) == dict:
        obj = sorted(obj.items(), key = sorter, reverse = True)
        check_threshold = obj[0][1]['score'] < threshold
    else:
        if enforce_detection:
            raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

        if type(insurance_file) == type(None):
            return base_img, img_region, 'restore'
        
        get_original_obj = True 

    if get_original_obj or check_threshold:
        obj = sorted(
            insurance_file.loc[file.loc[img_abspath, 'original'], 'retinaface_representation'].items(),
            key = sorter,
            reverse = True
        )
        info = 'from insurance model'
        
    resp = []

    for _, identity in obj:

        facial_area = identity["facial_area"]

        y = facial_area[1]
        h = facial_area[3] - y
        x = facial_area[0]
        w = facial_area[2] - x
        img_region = [x, y, w, h]

        #detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
        detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

        if align:
            landmarks = identity["landmarks"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            nose = landmarks["nose"]
            #mouth_right = landmarks["mouth_right"]
            #mouth_left = landmarks["mouth_left"]

            detected_face = postprocess.alignment_procedure(detected_face, right_eye, left_eye, nose)
            
        if padd:
            factor_0 = target_size[0] / detected_face.shape[0]
            factor_1 = target_size[1] / detected_face.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(detected_face.shape[1] * factor), int(detected_face.shape[0] * factor))
            detected_face = cv2.resize(detected_face, dsize)

            # Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - detected_face.shape[0]
            diff_1 = target_size[1] - detected_face.shape[1]
                
            detected_face = np.pad(detected_face, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
            
        detected_face = cv2.resize(detected_face, target_size)

        #normalizing the image pixels

        img_pixels = image.img_to_array(detected_face) #what this line doing? must?
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255 #normalize input in [0, 1]

        resp.append((img_pixels, img_region))

    if find_all:
        return resp, info

    return *resp[0], info

# dir processing with current model
def find(test_db_path, df_base # info 
    
    , face_detector = None         # detector model
    
    , insurance_file = None
    , enforce_detection = True     # detection settings
    , threshold = 0.3   

    , align = False                # postprocess
    , padd = False
    
    , model = None                 # recognizer model 
    , model_name = 'ArcFace' 
    , model_input_size = None

    , metric_name = 'cosine'       # metric 
    
    , rewrite = False              # other settings
    , progressbar_out = None
) -> pd.DataFrame:

    full_path = get_path_to_res_plk(test_db_path, model_name, align, padd)

    if not rewrite and os.path.exists(full_path):
        with open(full_path, "rb") as f:
            return pickle.load(f)
    
    #create res from dir
    
    df = df_base.copy()
    resp_obj = []
    indexs = []
    success = 0
    
    get = '%s_%s' % (model_name, metric_name)    
    indexs, size = collect_image_from_dir(test_db_path)
    file = load_face_object(test_db_path)
    
    if progressbar_out == None:
        progressbar_out = range(0, size)

    for it in progressbar_out:
        exact_path = os.path.abspath(indexs[it])
        original_path = os.path.abspath(file.loc[exact_path, 'original'])

        detected_face, face_region, info = preprocess_face(exact_path            
            , target_size = model_input_size
            , face_detector = face_detector           # detector model
            , enforce_detection = enforce_detection   # detection settings
            , threshold = threshold
            , file = file
            , insurance_file = insurance_file
            , align = align                           # postprocess
            , padd = padd
        )

        face_representation = model.predict(detected_face)[0].tolist()

        distances = []
    
        for _, instance in df.iterrows():
            source_representation = instance['representation']
            distance = findCosineDistance(source_representation, face_representation)
            distances.append(distance)
        
        df[get] = distances
        #df = df.drop(columns = ['%s_representation' % (model_name)])
        df = df.sort_values(by = [get], ascending = True)
        res = False
        
        if os.path.samefile(original_path, df.iloc[0].name):
            res = True
            success += 1

        resp_obj.append([face_representation, res, face_region, info, df])
        df = df_base.copy() #restore df for the next iteration


    output = pd.DataFrame(resp_obj, index = indexs, columns = ['%s_representation' % (model_name), 'res', 'area', 'info', 'data']), success, size, 100 * success / size 
        
    with open(full_path, "wb") as f:
        pickle.dump(output, f)
            
    return output


# visualize in ipynb
import ipywidgets as widgets
from IPython.display import display, clear_output

def visualize(test_db_path
    , qshow = 3
    , detector_backend = 'retinaface'
    , model_name = 'ArcFace'
    , metric_name = 'cosine'
    , align = False
    , padd = False):

    get = '%s_%s' % (model_name, metric_name)

    full_path = get_path_to_res_plk(test_db_path, model_name, align, padd)
    
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            df_test, sucsses, size, perc = pickle.load(f)
    else:
        raise ValueError('file " %s " is not exist' % (full_path))
    
    dim, _ = df_test.shape

    labelin = widgets.Label(value='Input')
    labelres = widgets.Label(value='    TRUE')
    labelout = widgets.Label(value='Output')
    labelpercent = widgets.Label(value = f'RES : {perc} %')

    slider = widgets.IntSlider(description ='Face', value = 2, max = dim, min = 1)
    
    disp_face = widgets.Image()
    disp_face_rec = widgets.Image()
    #disp_face_extract = widgets.Image()
    grid1 = widgets.GridspecLayout(1, qshow)
    grid2 = widgets.GridspecLayout(1, qshow)
    for j in range(qshow):
        grid1[0, j] = widgets.Label(value='d :')
        grid2[0, j] = widgets.Image()
    
    def update(change):
        if change['type'] == 'change' and change['name'] == 'value':
            series = df_test.iloc[slider.value - 1]
            flag, rec, df_info = series
            face_path = series.name
            x, y, w, h = rec
            
            face_rec = toBGR(functions.load_image(face_path))
            cv2.rectangle(face_rec, rec, (0, 255, 0), 5)
            labelres.value = '    FALSE'
            if flag:
                labelres.value = '    TRUE'
            disp_face.set_value_from_file(face_path)
            disp_face_rec.value = img_to_bytes(face_rec)
            #disp_face_extract.value = img_to_bytes(face[y : y + h, x : x + w])
            for j in range(qshow):
                grid1[0, j].value = f'{metric_name}: {df_info[get][j]}'
                grid2[0, j].set_value_from_file(df_info['identity'][j])

    slider.observe(update, names='value')
    display(widgets.VBox([slider, labelin, widgets.HBox([disp_face, disp_face_rec, labelres]), labelout, grid1, grid2, labelpercent]))
    slider.value = 1

if __name__ == '__main__':
    print('module by Nikita Shulgin')