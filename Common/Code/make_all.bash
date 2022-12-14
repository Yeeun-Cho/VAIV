source ~/anaconda3/etc/profile.d/conda.sh
conda activate YEenv

for year in {2022..2019..-1}
do
    python /home/ubuntu/2022_VAIV_Cho/VAIV/Common/Code/make_all.py --yolo -y $year &
done
