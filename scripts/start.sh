alg=$1
env=$2
map=$3

if [ "$env" == "metaworld" ]; then
    if [ "$map" == "mt10" ]; then
        replay_buffer_capacity="1000000"
        replay_buffer_batch_size="1280"
        num_train_steps="1500100"
    elif [ "$map" == "mt50" ]; then
        replay_buffer_capacity="10000000"
        replay_buffer_batch_size="6400"
        num_train_steps="1500100"
    else
        echo "Error map_name in $env: $map"
        exit 1
    fi
elif [ "$env" == "gym_extensions" ]; then
    if [ "$map" == "halfcheetah_gravity-mt5" ]; then
        replay_buffer_capacity="500000"
        replay_buffer_batch_size="640"
        num_train_steps="800100"
    elif [ "$map" == "halfcheetah_body-mt8" ]; then
        replay_buffer_capacity="1000000"
        replay_buffer_batch_size="1024"
        num_train_steps="800100"
    else
        echo "Error map_name in $env: $map"
        exit 1
    fi
else
    echo "Error env_name: $env"
    exit 1
fi

if [ "$fix_or_random" == "rand" ]; then
    env_random_goal="True"
else
    env_random_goal="False"
fi

bash scripts/alg/$alg.sh $env $map $replay_buffer_capacity $replay_buffer_batch_size $num_train_steps