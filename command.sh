# conda activate vibe-env

:<<! 
-------------------- elderly side ----------------------
    using side epoch model

for i in {1..300}
do
    python test.py --vid_file /media/user/WD-SN750SE/chunhua/fall-prevention-data/sit-to-stand/cs-side-$i.mp4  --output_folder /home/user/ply/data/five_sts/keypoints_3d/vibe/side/multi_person/ --no_render
done
!


:<<!
--------------------- elderly front ---------------------
    using side+front epoch model

for i in {1..300}
do
    python test.py --vid_file /media/user/WD-SN750SE/chunhua/fall-prevention-data/sit-to-stand/cs-front-$i.mp4  --output_folder /home/user/ply/data/five_sts/keypoints_3d/vibe/side/multi_person/ --no_render
done
!


:<<!
---------------------- students side ------------------

for i in {1..300}
do
    python test.py --vid_file /media/user/WD-SN750SE/chunhua/fall-prevention-data/sit-to-stand/students/stu-cs-side-$i.mp4  --output_folder /home/user/ply/data/five_sts/keypoints_3d/vibe/side/multi_person/ --no_render
done
!


:<<!
----------------------- small batch test -------------------
!


for i in {178..178}

do
    python test.py --vid_file /media/user/WD-SN750SE/chunhua/fall-prevention-data/sit-to-stand/cs-side-$i.mp4  --output_folder ./output/cs-side-$i/ --no_render
done



:<<!
for i in {225..225}
do
    python test.py --vid_file /media/user/WD-SN750SE/chunhua/fall-prevention-data/sit-to-stand/cs-side-$i.mp4  --output_folder ./output/cs-side-$i/ --display --sideview --no_render
done
!

:<<!

mkdir ./output/cs-side-$i
/home/user/ply/data/five_sts/keypoints_3d/vibe/side/multi_person/

!
