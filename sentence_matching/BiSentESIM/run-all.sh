nohup python -u driver/Train.py  --thread 1 --gpu 7 --config msr/default.cfg > msr.log 2>&1 &
nohup python -u driver/Train.py  --thread 1 --gpu 4 --config quora/default.cfg  > quora.log 2>&1 &
nohup python -u driver/Train.py  --thread 1 --gpu 6 --config sci/default.cfg   > sci.log 2>&1 &
nohup python -u driver/Train.py  --thread 1 --gpu 7 --config sick/default.cfg  > sick.log 2>&1 &
nohup python -u driver/Train.py  --thread 1 --gpu 5 --config snli/default.cfg  > snli.log 2>&1 &

