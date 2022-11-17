
curl -L -o joint_data.rda https://osf.io/download/zyprg
mv ./joint_data.rda ../data/joint_data.rda
cd ../data/
mkdir langs
Rscript ./../scripts/seperate_langs.R

