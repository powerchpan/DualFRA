import joblib
import numpy as np
import json
import matplotlib.pyplot as plt


def draw_image(people_id, y):

    y = np.array(y)
    i = 0
    y_shorter = []
    while i<len(y):
        y_shorter.append(y[i])
        i+=5
    y_shorter = np.array(y_shorter)
    length = len(y_shorter)
    x = np.zeros((length))
    for i in range(length):
        x[i] = i+1
    plt.plot(x, y_shorter, 'bo--', alpha=0.5, linewidth=1, label=str(people_id))  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.legend()
    plt.savefig('/home/user/ply/data/five_sts/keypoints_3d/side/'+str(people_id)+'.jpg')
    plt.show()





raw_data_dir = '/home/user/ply/data/five_sts/keypoints_3d/hospital_20230410/raw/front_good/cs-front-'
save_path = '/home/user/ply/data/five_sts/keypoints_3d/hospital_20230410/target_keypoints/front_good/162.npy'
save_multiPeople_keypoints_dir = '/home/user/ply/data/five_sts/keypoints_3d/hospital_20230410/keypoints_3d_multiPersons/front_good/'
i = 162
while i<163:
    #end_path = ''+str(i)+'/vibe_output.pkl'
    #raw_data_path = raw_dir+end_path
    if((i==23)|(i==60)|(i==74)|(i==77)|(i==85)|(i==87)|(i==102)|(i==114)|(i==115)|(i==117)|(i==118)|(i==147)):
            print(i)
            i+=1
            continue
    raw_data_end_path = str(i)+'/vibe_output.pkl'
    raw_data_path = raw_data_dir+raw_data_end_path
    print(i)
    output = joblib.load(raw_data_path)
    people_num = len(output)
    print(output.keys())
    multiPeople_keypoints = []
    save_multiPeople_keypoints_end_path = str(i)+'.npy'
    save_multiPeople_keypoints_path = save_multiPeople_keypoints_dir + save_multiPeople_keypoints_end_path
    for j in output.keys():
        keypoints_3d = output[j]['joints3d']
        keypoints_3d_25 = keypoints_3d[:,0:25,:] 
        multiPeople_keypoints.append(keypoints_3d_25)
        nose_y = keypoints_3d_25[:, 0, 1]
        draw_image(j, nose_y)
    np.save(save_multiPeople_keypoints_path, multiPeople_keypoints) 
    i+=1
keypoints_3d = output[71]['joints3d']
keypoints_3d_25 = keypoints_3d[:,0:25,:] 
np.save(save_path, keypoints_3d_25)  
    #keypoints_3d = output[0]['joints3d']
    #keypoints_3d_25 = keypoints_3d[:,0:25,:] 
    #nose_y = keypoints_3d_25[:, 0, 1]
    #draw_image(0, nose_y)   
    #for i in range(people_num):
        #keypoints_3d = output[2]['joints3d']
        #keypoints_3d_25 = keypoints_3d[:,0:25,:]
        #nose_y = keypoints_3d_25[:, 0, 1]
        #draw_image(i, nose_y)
    #print(people_num)
    #keypoints_3d = output[1]['joints3d']
    #keypoints_3d_25 = keypoints_3d[:,0:25,:]
    #nose_y = keypoints_3d_25[:, 0, 1]
    
    #print(keypoints_3d_25.shape)
    #for k, v in output[1].items():
        
        

