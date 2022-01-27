import numpy as np
import os
from numba import njit
import queue
import _thread
from pathlib import Path

@njit()
def farthest_point_sample_njit(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,)) # clustering center
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    point = point[centroids.astype(np.int32)]
    return point
def worker(q,input_folder,output_folder):
    while q.empty()==False:
        xyz_name=q.get()
        print(xyz_name)
        root='_fps.xyz'
        output_name=xyz_name.replace(input_folder,output_folder)
        output_name=output_name.replace('.xyz',root)
        ### Check if it is processed
        flag=Path(xyz_name)
        ###
        if flag.exists():
            # q.task_done()
            continue
        else:
            xyz=np.loadtxt(xyz_name)
            xyz_fps=farthest_point_sample_njit(xyz,350000)

            np.savetxt(output_name,xyz_fps)
            q.task_done()
    
def main():
    input_folder='E:/Ender/data/backpack/normalized_subsets'
    output_folder='E:/Ender/data/backpack/normalized_subsets_fps'
    try:
        os.mkdir(output_folder)
    except:
        pass

    xyz_dates=os.listdir(input_folder)
    for i in range(len(xyz_dates)):
        sub_dir=input_folder+'/'+xyz_dates[i]
        try:
            os.mkdir(output_folder+'/'+xyz_dates[i])
        except: pass
        las_list=os.listdir(sub_dir)
        q=queue.Queue()
        for j in range(len(las_list)):
            # if las_list[j].endswith('subset2.las') or las_list[j].endswith('subset3.las') or las_list[j].endswith('subset4.las') :
            xyz_name=sub_dir+'/'+las_list[j]
            q.put(xyz_name)
    for workers_num in range(16):
        _thread.start_new_thread(worker,(q,input_folder,output_folder))
    q.join()
if __name__ == '__main__':
    main()