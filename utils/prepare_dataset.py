import json
import glob
import os
from natsort import natsorted
import shutil
import math
import numpy as np
import cv2 as cv
from plyfile import PlyData, PlyElement
from datetime import datetime
from absl import flags, logging, app

_ZEROANGLES=1E-10

FLAGS = flags.FLAGS
flags.DEFINE_string('data_folder', None, 'Path to dataset. The script expects a folder with scene subfolders.')
flags.DEFINE_string('output_folder', None, 'output')
flags.DEFINE_string('models_path', None, 'Path to PLY models')
flags.DEFINE_string('obj_map', None, 'Path to a json file with the mapping between the object IDs and the class names')
flags.DEFINE_spaceseplist('scenes', None, 'Space-separated list of which scenes you want to process. Default will process all scenes. E.g --scenes "000001 000005"')
flags.DEFINE_integer('digits', 6, '')
# def calculate_bbox(ply_model): using DOPE coordinate system

#     x = ply_model['vertex']['x']
#     y = ply_model['vertex']['y']
#     z = ply_model['vertex']['z']
#     xmin = np.min(x)
#     xmax = np.max(x)
#     ymin = np.min(y)
#     ymax = np.max(y)
#     zmin = np.min(z)
#     zmax = np.max(z)

#     centroid = [np.sum(x)/x.size, np.sum(y)/y.size, np.sum(z)/z.size]
#     print('@@@@ centroid', centroid)
#     #return np.array([[xmin, ymax, zmax],[xmax, ymax, zmax], [xmax, ymin, zmax],[xmin, ymin, zmax],[xmin, ymax, zmin],[xmax,ymax,zmin],[xmax, ymin, zmin], [xmin,ymin,zmin], centroid])
#     return np.array([[xmin, ymin, zmax], \
#                      [xmax, ymin, zmax], \
#                     [xmax, ymax, zmax], \
#                     [xmin,ymax,zmax], \
#                     [xmin,ymin,zmin], \
#                     [xmax, ymin, zmin], \
#                     [xmax,ymax,zmin], \
#                     [xmin,ymax,zmin], \
#                     centroid])
# def calculate_projected_cuboid(bbox, R_vector, t_vector, camera_matrix):
#     result,_= cv.projectPoints(bbox, R_vector, t_vector, cameraMatrix=camera_matrix,distCoeffs=None)
#     return result
def read_json(filepath):
    with open(filepath, 'rb') as f:
        data = json.load(f)
    return data 

def write_json(filepath, data):
    fd = open(filepath, 'w')
    json.dump(data, fd, indent=4)
    fd.close()
    
def calculate_bbox(ply_model): #using EPOS coordinate system

    x = ply_model['vertex']['x']
    y = ply_model['vertex']['y']
    z = ply_model['vertex']['z']
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)

    centroid = [np.sum(x)/x.size, np.sum(y)/y.size, np.sum(z)/z.size]

    return np.array([ [xmax, ymax, zmin],[xmax, ymax, zmax],[xmin, ymax, zmax], [xmin, ymax, zmin], [xmax, ymin, zmin], [xmax,ymin, zmax], [xmin, ymin, zmax], [xmin, ymin, zmin], centroid])

def calculate_projected_cuboid(bbox, R_matrix, t_matrix, camera_matrix):
    RT = np.identity(4)
    RT[:3,:] = np.column_stack([R_matrix, t_matrix])
    logging.info('RT'+ str(RT))
    logging.info('BB'+str(bbox))
    tpoints = [ camera_matrix @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in bbox] 
    
    tpoints = np.array([i/i[-1] for i in tpoints]) #divide with last element to convert from homogeneous coordinates
    logging.info(str(tpoints))
    tpoints_list = tpoints.tolist()
    # eliminate homogeneous coordinate and fix the list to be consistent with the format we want 
    projected_cuboid = []
    for item in tpoints_list:
        vertex = item[0] + item[1]
        projected_cuboid.append(vertex)
    
    logging.info(str(projected_cuboid) + ''+ str(type(projected_cuboid)))
    return projected_cuboid
# or 
# RT = np.identity(4)
# RT[:3,:] = np.column_stack([Rest,Test])


# result,_= cv.projectPoints(model,method_results['0']['rvec'],method_results['0']['tcev'],cameraMatrix=K,distCoeffs=None)

# # or
## maybe i should follow that instead
# tpoints = [ K @ RT[:3,:] @ np.append(i,[1]).reshape(4,1) for i in model] 

# # remember tpoints are in 2D homogenous coordinates. In order to obtain the cartesian 2D coordinates we need to perform perspective devision with w
# tpoints = np.array([i/i[-1] for i in tpoints])

def vec2quat(rot):

    qo = np.zeros((4,))
    th = math.sqrt(math.pow(rot[0], 2)+math.pow(rot[1], 2)+math.pow(rot[2], 2))
    if th > _ZEROANGLES:
        hth = th*0.5
        s = math.sin(hth)
        qo[0] = math.cos(hth)
        s = s/th
        qo[1] = rot[0]*s
        qo[2] = rot[1]*s
        qo[3] = rot[2]*s

    else:
        qo[0] = 1.0
        qo[1] = qo[2] = qo[3] = 0.0

    return qo

# get image id from image name as string to use as key for dictionary lookup
def get_image_id(img_path):
    return os.path.splitext(os.path.basename(img_path))[0]

def get_folders(scenes, data_folder):
    if scenes is None:    
        folders = natsorted(glob.glob(data_folder+"/*"))
    else:
        folders = []
        scenes = natsorted(scenes)
        #import pdb; pdb.set_trace
        for s in scenes:
            
            logging.info(s)
            for p in glob.glob(data_folder +"/*"+s):
                folders.append(p)

    return folders


# data_folder = '/home/foto1/linux_part/athena/isaac/dope_training/dope_training/train_primesense'
# output_folder = '/home/foto1/linux_part/athena/isaac/dope_training/dope_training/train_data4'
# model_1_path = '/home/foto1/linux_part/athena/isaac/dope_training/dope_training/phase_II_objects/obj_000001.ply'
# model_2_path = '/home/foto1/linux_part/athena/isaac/dope_training/dope_training/phase_II_objects/obj_000002.ply'
# debug = False

def main(argv):
    
    obj_map = read_json(FLAGS.obj_map)
    # for k in obj_map.keys():
    #     class_folder = os.path.join(FLAGS.output_folder, obj_map[k])
    #     if not os.path.exists(class_folder):
    #         logging.info('Creating '+str(class_folder))
    #         os.makedirs(class_folder)
    bb = []
    models = natsorted(glob.glob(FLAGS.models_path+"/*.ply"))
    for p in models:
        with open(p, 'rb') as f:
            logging.info('Reading model '+ p)
            model = PlyData.read(f)

    # with open(model_2_path, 'rb') as f:
    #     model_2 = PlyData.read(f)

        bb.append(calculate_bbox(model))
    #bb2 = calculate_bbox(model_2)

    folders = get_folders(FLAGS.scenes, FLAGS.data_folder)
    #scenes = natsorted(glob.glob(FLAGS.data_folder+"/*")) 
    # load models in a list, calculate their bounding boxes. 
    # then when iterating the image folders, check objid of items every time and choose the appropriate bounding box to project on the image
    image_names = {} # since all images from every scene need to be saved in a single folder, keep a mapping of the new image basenames to the original filepaths
    logging.info(folders)

    # num_images = 0
    # for s in folders:
    #     images = natsorted(glob.glob(s+"/rgb/*"))
    #     num_images = num_images + len(images)

    # logging.info('Total number of training images: ' +str(num_images))

    if not os.path.exists(FLAGS.output_folder):
        logging.info('Creating output folder ' + FLAGS.output_folder)
        os.mkdir(FLAGS.output_folder)

    count = 0
    for s in folders:
        #import pdb; pdb.set_trace()
        camera_filepath = os.path.join(s, 'scene_camera.json')
        #camera_data = read_json(camera_filepath)
        camera_data_raw = read_json(camera_filepath)
        cam_key = [*camera_data_raw.keys()][0] # unpack keys dictionary to a list and get first element 
        camera_data = np.reshape(np.array(camera_data_raw[cam_key]["cam_K"]), (3,3)) 
        logging.info('Loaded camera data from' + camera_filepath)
        logging.info(str(camera_data))

        gt_file = os.path.join(s, 'scene_gt.json')
        gt_data = read_json(gt_file)
        logging.info('Loaded GT data from' + gt_file)

        #print(gt_data)

        gt_info_file = os.path.join(s, 'scene_gt_info.json')
        gt_info_data = read_json(gt_info_file)
        logging.info('Loaded GT info data from' + gt_info_file)
        scene_id = os.path.splitext(s)[0].split("/")[-1]
        out_scene = os.path.join(FLAGS.output_folder, scene_id) 
        logging.info("Creating output scene folder")
        os.makedirs(out_scene)
        #print(gt_info_data)

        rgb_img_paths = natsorted(glob.glob(s+"/rgb/*"))
        for path in rgb_img_paths:
            #print(idx, key)
            object_list = []
            key = get_image_id(path)
            logging.info('key'+ str(key))
            #import pdb; pdb.set_trace()
            if key not in gt_data.keys(): # to cover edge case where the key is of the form eg "000001" i.e. same as image base name and not "1"
                key = str(int(key))
            for i, object in enumerate(gt_data[key]):
                R_matrix = np.reshape(np.array(object["cam_R_m2c"]), (3,3))
                logging.info('R'+ str(R_matrix))
                rotation_vector,_ = cv.Rodrigues(R_matrix)
                quaternion = vec2quat(rotation_vector)
                #print('quaternion', quaternion)

                t_matrix = object["cam_t_m2c"]
                logging.info('t matrix'+str(t_matrix))
                #translation_vector,_ = cv.Rodrigues(t_matrix)

                #print('translation vector', translation_vector)

                obj_id = object["obj_id"]
                projected_cuboid = calculate_projected_cuboid(bb[obj_id-1], R_matrix, np.array(t_matrix), camera_data)
                class_name = obj_map[str(obj_id)]
                # if obj_id == 1:
                #     class_name = 'Makita_DFT_obj_id_1'
                #     projected_cuboid = calculate_projected_cuboid(bb1, R_matrix, np.array(t_matrix), camera_data)
                # elif obj_id == 2:
                #     class_name = 'Windows_control_panel_1_obj_id_2'
                #     projected_cuboid = calculate_projected_cuboid(bb2, R_matrix, np.array(t_matrix), camera_data)

                if key not in gt_info_data.keys(): # to cover edge case where the key is of the form eg "000001" i.e. same as image base name and not "1"
                    info_key = str(int(key))
                else:
                    info_key = key
                visibility = gt_info_data[info_key][i]["visib_fract"]
                object_list.append({
                                "class": class_name,
                                "visibility": visibility,
                                "location": t_matrix, 
                                "quaternion_xyzw": quaternion.tolist(),
                                "projected_cuboid": projected_cuboid #should be a list of vertices
                            })
            annotation = {"camera_data": {},
                        "objects": object_list
                        }
            
            #import pdb; pdb.set_trace()
            new_img_basename = (len(str(FLAGS.digits))-len(str(count)))*'0' + str(count)
            new_img_path = os.path.join(out_scene, new_img_basename + ".png")
            logging.info('Copying '+path+' to '+new_img_path)

            image_names[new_img_basename] = path

            
            shutil.copy2(path, new_img_path)

            #print('Annotation for image', rgb_img_paths[idx], ":", annotation)
            annot_file = os.path.join(out_scene, new_img_basename + ".json")
            logging.info('Saving annotation file '+ annot_file)
            write_json(annot_file, annotation)
                
            
            count = count + 1

        write_json(os.path.join(FLAGS.output_folder, 'image_mapping.json'), image_names)

#print(image_names)

if __name__ == '__main__':
    app.run(main)
