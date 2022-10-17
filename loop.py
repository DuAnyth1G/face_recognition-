import facepreprocessing as fp
import pandas as pd
import itertools
import my_bar
import pickle
import os

path = 'W:\\FaceMask\\'
work_dirs = list(fp.work_dir_generator(path))

def set_paths_to_originals(dir):
    image_test_list, size = fp.collect_image_from_dir(dir)
    image_orig_list,    _ = fp.collect_image_from_dir('0_ORIGINAL_PNG_250_200\\0_ORIGINAL_PNG_250_200\\')

    if size > 100:
        return list(itertools.chain.from_iterable([[os.path.abspath(it)] * 10 for it in sorted(image_orig_list, key = lambda x: int(fp.get_file_name(x)))]))

    if size < 100:
        n = os.path.abspath(dir).split('\\')[-1]
        return [os.path.abspath(it) for it in image_orig_list if int(n) == int(fp.get_file_name(it))] * 18
    
    return list(map(os.path.abspath, image_orig_list))

def do_work(dir):
    df_base = fp.load_original(
        db_path = dir,
        model_name = 'retinaface'
    )

    output = pd.DataFrame(
        data = [
            [dic, os.path.abspath(orig)] for dic, orig in zip(
                df_base['retinaface_representation'].to_list()
                ,
                set_paths_to_originals(dir)
            )
        ],
        index = list(map(os.path.abspath, df_base['identity'])),
        columns=['retinaface_representation', 'original']
    )

    file_name = 'representations_retinaface_new_with_insurance.pkl'
    full_path = dir + file_name 
    with open(full_path, "wb") as f:
        pickle.dump(output, f)


with my_bar.progress as progress:

    main_task = progress.add_task(f'[cyan]Retinaface reform data          ...')
    #work_dirs = ['W:\\ETU\\third_term\\KMIL\\0_ORIGINAL_PNG_250_200\\0_ORIGINAL_PNG_250_200\\']

    for dir in progress.track(sequence = work_dirs, task_id = main_task):
        
        progress.print(dir.replace('\\', '/'))
    
        do_work(dir)
    
    #progress.remove_task(task_id = main_task)