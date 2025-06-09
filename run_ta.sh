num=20
# echo ">>>>> randomly assigned tasks  >>>>>"
# echo ">>>>> randomly assigned tasks  >>>>>" > log_ta.txt
# for (( k = 0; k < $num / 5; k++ ))
# do
#     for (( i=0; i < 5; i++ ))
#     do
#         echo robot seed: $k, task seed: $i
#         python run_maituan_ta.py --assigner random --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#     done
# done

# echo ">>>>> closer port first  >>>>>"
# echo ">>>>> closer port first  >>>>>" >> log_ta.txt
# for (( k = 0; k < $num / 5; k++ ))
# do
#     for (( i=0; i < 5; i++ ))
#     do
#         echo robot seed: $k, task seed: $i
#         python run_maituan_ta.py --assigner closer --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#     done
# done

# echo ">>>>> farther port first  >>>>>"
# echo ">>>>> farther port first  >>>>>" >> log_ta.txt
# for (( k = 0; k < $num / 5; k++ ))
# do
#     for (( i=0; i < 5; i++ ))
#     do
#         echo robot seed: $k, task seed: $i
#         python run_maituan_ta.py --assigner farther --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#     done
# done

# echo ">>>>> alpha 0.23363  >>>>>"
# echo ">>>>> alpha 0.23363 >>>>>" >> log_ta.txt
# for (( k = 0; k < $num / 5; k++ ))
# do
#     for (( i=0; i < 5; i++ ))
#     do
#         echo robot seed: $k, task seed: $i
#         python run_maituan_ta.py --assigner alpha --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#     done
# done

# for (( cfg = 0; cfg < 3; cfg++ ))
# do
#     echo ">>>>> rl cfg $cfg >>>>>"
#     echo ">>>>> rl cfg $cfg >>>>>" >> log_ta.txt
#     for (( k = 0; k < $num / 5; k++ ))
#     do
#         for (( i=0; i < 5; i++ ))
#         do
#             echo robot seed: $k, task seed: $i
#             python run_maituan_ta.py --assigner mpc --cfg $cfg --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#         done
#     done
# done

# for (( model = 3; model < 6; model++ ))
# do
#     echo ">>>>> rl model $model >>>>>"
#     echo ">>>>> rl model $model >>>>>" >> log_ta.txt
#     for (( k = 0; k < $num / 5; k++ ))
#     do
#         for (( i=0; i < 5; i++ ))
#         do
#             echo robot seed: $k, task seed: $i
#             python run_maituan_ta.py --assigner mpc --model $model --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#         done
#     done
# done

# for (( model = 6; model < 8; model++ ))
# do
#     echo ">>>>> rl model $model >>>>>"
#     echo ">>>>> rl model $model >>>>>" >> log_ta.txt
#     for (( k = 0; k < $num / 5; k++ ))
#     do
#         for (( i=0; i < 5; i++ ))
#         do
#             echo robot seed: $k, task seed: $i
#             python run_maituan_ta.py --assigner mpc --model $model --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#         done
#     done
# done

# for (( model = 8; model < 10; model++ ))
# do
#     echo ">>>>> rl model $model >>>>>"
#     echo ">>>>> rl model $model >>>>>" >> log_ta.txt
#     for (( k = 0; k < $num / 5; k++ ))
#     do
#         for (( i=0; i < 5; i++ ))
#         do
#             echo robot seed: $k, task seed: $i
#             python run_maituan_ta.py --assigner mpc --model $model --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#         done
#     done
# done

# for (( model = 10; model < 12; model++ ))
# do
#     echo ">>>>> rl model $model >>>>>"
#     echo ">>>>> rl model $model >>>>>" >> log_ta.txt
#     for (( k = 0; k < $num / 5; k++ ))
#     do
#         for (( i=0; i < 5; i++ ))
#         do
#             echo robot seed: $k, task seed: $i
#             python run_maituan_ta.py --assigner mpc --model $model --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#         done
#     done
# done

# for (( model = 12; model < 14; model++ ))
# do
#     echo ">>>>> rl model $model >>>>>"
#     echo ">>>>> rl model $model >>>>>" >> log_ta.txt
#     for (( k = 0; k < $num / 5; k++ ))
#     do
#         for (( i=0; i < 5; i++ ))
#         do
#             echo robot seed: $k, task seed: $i
#             python run_maituan_ta.py --assigner mpc --model $model --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#         done
#     done
# done

# echo ">>>>> tpts + randomly assigned tasks  >>>>>"
# echo ">>>>> tpts + randomly assigned tasks  >>>>>" >> log_ta.txt
# for (( k = 0; k < $num / 5; k++ ))
# do
#     for (( i=0; i < 5; i++ ))
#     do
#         echo robot seed: $k, task seed: $i
#         python run_maituan_ta.py --router tpts --assigner random --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#     done
# done

# echo ">>>>> tpts + closer port first  >>>>>"
# echo ">>>>> tpts + closer port first  >>>>>" >> log_ta.txt
# for (( k = 0; k < $num / 5; k++ ))
# do
#     for (( i=0; i < 5; i++ ))
#     do
#         echo robot seed: $k, task seed: $i
#         python run_maituan_ta.py --router tpts --assigner closer --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#     done
# done

# echo ">>>>> tpts + farther port first  >>>>>"
# echo ">>>>> tpts + farther port first  >>>>>" >> log_ta.txt
# for (( k = 0; k < $num / 5; k++ ))
# do
#     for (( i=0; i < 5; i++ ))
#     do
#         echo robot seed: $k, task seed: $i
#         python run_maituan_ta.py --router tpts --assigner farther --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
#     done
# done

echo ">>>>> tptsR + randomly assigned tasks  >>>>>"
echo ">>>>> tptsR + randomly assigned tasks  >>>>>" >> log_ta.txt
for (( k = 0; k < $num / 5; k++ ))
do
    for (( i=0; i < 5; i++ ))
    do
        echo robot seed: $k, task seed: $i
        python run_maituan_ta.py --router tpts_r --assigner random --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
    done
done

echo ">>>>> tptsR + closer port first  >>>>>"
echo ">>>>> tptsR + closer port first  >>>>>" >> log_ta.txt
for (( k = 0; k < $num / 5; k++ ))
do
    for (( i=0; i < 5; i++ ))
    do
        echo robot seed: $k, task seed: $i
        python run_maituan_ta.py --router tpts_r --assigner closer --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
    done
done

echo ">>>>> tptsR + farther port first  >>>>>"
echo ">>>>> tptsR + farther port first  >>>>>" >> log_ta.txt
for (( k = 0; k < $num / 5; k++ ))
do
    for (( i=0; i < 5; i++ ))
    do
        echo robot seed: $k, task seed: $i
        python run_maituan_ta.py --router tpts_r --assigner farther --task-seed $i --robot-seed $k | tail -n 1 >> log_ta.txt
    done
done