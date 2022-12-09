
curl -L -o joint_data.rda https://osf.io/download/zyprg
mv ./joint_data.rda ../data/joint_data.rda
cd ../data/
mkdir langs_l1
cd ../scripts/

curl -L -o joint_data_l2_trimmed.rda https://osf.io/download/bf2q9
mv ./joint_data_l2_trimmed.rda ../data/joint_data_l2_trimmed.rda
cd ../data/
mkdir langs_l2

Rscript ./../scripts/seperate_langs.R