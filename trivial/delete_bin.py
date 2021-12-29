import os
path='H:/Ender/banding/20210605\Gimbal\gimbal_afternoon\100208_20200605GIMBAL0250_2021_06_05_18_51_25'
flist=os.listdir(path)
for i in flist:
    if i.endswith('.bin'):
        f=path+'/'+i
        os.remove(f)
        
