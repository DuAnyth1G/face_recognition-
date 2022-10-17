import facepreprocessing as fp

path = 'W:/FaceMask/'
work_dirs = list(fp.work_dir_generator(path))

from retinaface import RetinaFace
from retinaface.commons import postprocess

face_detector = RetinaFace.build_model()

import my_bar
from time import sleep
import pickle, os


def retina_spec_to_plk(
    dir,
    face_detector_model = face_detector,
    progressbar_out = None
    ) :

    if dir[-1] not in '\\/':
       dir += "\\"

    #images, size = fp.collect_image_from_dir(dir)
    
    representations = []
    
    if progressbar_out == None:
        progressbar_out = range(0, size)

	# loop
    for img in progressbar_out:
        
        instance = []
        instance.append(img)

        obj = RetinaFace.detect_faces(img
            , model = face_detector_model
            , threshold = 0.1
            )

        if type(obj) == dict:
            instance.append(obj)
        else:
            instance.append('no_face')

        representations.append(instance)

    file_name = 'representations_retinaface.pkl'
    full_path = dir + file_name 
    with open(full_path, "wb") as f:
        pickle.dump(representations, f)




if __name__ == '__main__':

    with my_bar.progress as progress:

        main_task = progress.add_task(f'[cyan]Analysis retinaface in          ...')


        for dir in progress.track(sequence = work_dirs, task_id = main_task):

            progress.print(dir.replace('\\', '/'))
            progress.update(task_id = main_task, description = f'[cyan]Analysis retinaface in {os.path.basename(dir[:-1])}...')
            task_id = progress.add_task(f'[cyan]Working on dir')
            
            img_list, size = fp.collect_image_from_dir(dir)
            retina_spec_to_plk(
                dir,
                progressbar_out = progress.track(sequence = img_list, task_id = task_id)
            )

            progress.remove_task(task_id)
