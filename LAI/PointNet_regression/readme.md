The codes in this folder are revised from yanx27's repo: https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
I convert the classifcation network into regression network.
- Use farthest point sampling to extract the same amount of points from each plot collected on different dates
- Use train_test_split.ipynb to prepare the json files for 10 fold cross validation. You need to get the plot id in each fold from ./Check_variety.ipynb
- There are various versions for training. I used train_test at the end. Please see the parse_args function to see the arguments for this code. 
An exampple:python train_test.py --batch_size 32 --model pointnet2_reg_msg --data HIPS_2021 --log_dir ma_log --fold 5 --gpu 0 --optimizer SGDM --epoch 500

The data preprocessing follows the flow chart below: \
<img src="https://github.com/Enderrrr-TW/Purdue_every_life/blob/main/LAI/PointNet_regression/methodology.png" width="400"> 
