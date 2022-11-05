source ~/anaconda3/etc/profile.d/conda.sh
conda activate YEenv

for year in {2022..2006..-1}
do
    python /home/ubuntu/2022_VAIV_Cho/VAIV/Common/Code/make_all.py --yolo -y $year &
    python /home/ubuntu/2022_VAIV_Cho/VAIV/Common/Code/make_all.py --cnn -y $year &
done
